// Run only inside the browser runtime's persistent JavaScript tool, with a claimed
// authenticated tab. Browser methods below are the documented browser-client API.
import { mkdir, writeFile, realpath } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import { platform } from 'node:os';
import { createHash, randomUUID } from 'node:crypto';
import path from 'node:path';

const CLOSE_PATH = 'M4.863 17.863L10.726 12 4.863 6.137';
// DOM scope setup shares this bounded dispatch budget. The implicit 3-second
// default has expired before dispatch on the live Brave extension connection.
// This changes read deadlines only; no action is retried or identity bypassed.
const DOM_READ_OPTIONS = Object.freeze({timeoutMs:10000});
// Acquisition time is distinct from the scanner's unchanged source-age gates.
// One snapshot period leaves the existing 900-second drawer gate authoritative.
const CAPTURE_DEADLINE_MS = 600000;
const REQUIRED = ['Working', 'Time Created', 'Total Profit (USDT)',
  'Invested Margin (USDT)', 'Matched Profit (USDT)', 'Positions',
  'Pending Order', 'Grid Details', 'Strategy Number'];

export function parseDrawer(text, symbol, created) {
  const normalized = text.replaceAll('\u00a0', ' ').trim();
  if (!normalized.startsWith(`${symbol}\n`)) throw new Error('drawer symbol mismatch');
  for (const marker of REQUIRED) {
    if (!normalized.includes(marker)) throw new Error(`incomplete drawer: ${marker}`);
  }
  const historyPresent = /^Order History$/m.test(normalized)
    || (/^History$/m.test(normalized) && normalized.includes('Total Matched Profit')
      && normalized.includes('Total Matched Trades'));
  if (!historyPresent) throw new Error('incomplete drawer: matched history');
  const ids = [...normalized.matchAll(/Strategy Number\s+(\d+)/g)];
  const times = [...normalized.matchAll(/Time Created\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})/g)];
  if (ids.length !== 1 || times.length !== 1 || times[0][1] !== created) {
    throw new Error('ambiguous strategy or changed deployment time');
  }
  return { strategyId: ids[0][1], createdAt: times[0][1] };
}

export function validateRoster(count, rows) {
  if (!Number.isSafeInteger(count) || count < 1) throw new Error('no proven active roster');
  if (rows.length !== count) throw new Error(`PARTIAL roster: ${rows.length}/${count}`);
  const symbols = rows.map(r => r.symbol);
  if (new Set(symbols).size !== count) throw new Error('duplicate active symbol');
  for (const row of rows) {
    if (!/^[A-Z0-9]+USDT$/.test(row.symbol) || row.status !== 'Working'
      || !/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/.test(row.created)) {
      throw new Error('invalid Working identity');
    }
  }
  return rows;
}

async function rosterView(tab) {
  // One read takes the selected tabs and rendered Virtual Table rows from the
  // same DOM instant. The exact table role/label was verified on the live page.
  const view = await readDOM(()=>tab.playwright.getByRole('tab').evaluateAll(elements=>{
    const tables=[...document.querySelectorAll('[role="table"][aria-label="Virtual Table"]')];
    const rows=tables.length===1?[...tables[0].querySelectorAll('[role="row"]')]:[];
    return {tabs:elements.map(e=>({text:(e.textContent||'').trim(),selected:e.getAttribute('aria-selected')})),
      tableCount:tables.length,rows:rows.flatMap(row=>{
        const cells=[...row.querySelectorAll('[role="cell"]')];
        if (!cells.length) return [];
        const symbol=(cells[0].textContent||'').trim().match(/^([A-Z0-9]+USDT)\s+Perp/);
        const created=(cells[1]?.textContent||'').trim().replace(/(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})/,'$1 $2');
        return [{symbol:symbol?.[1]||'',created,status:(cells[14]?.textContent||'').trim(),
          investment_text:(cells[2]?.textContent||'').trim()}];
      })};
  },undefined,DOM_READ_OPTIONS));
  const {tabs,rows}=view;
  if (view.tableCount!==1) throw new Error('Working table unavailable or ambiguous');
  const counts = tabs.map(t => /^UM Grid \((\d+)\)$/.exec(t.text)).filter(Boolean);
  if (counts.length !== 1) throw new Error('authenticated UM Grid count unavailable');
  for (const name of [counts[0][0], 'Running']) {
    const matching=tabs.filter(t=>t.text===name);
    if (matching.length!==1 || matching[0].selected!=='true') {
      throw new Error(`required tab is not selected or ambiguous: ${name}`);
    }
  }
  validateRoster(rows.length,rows);
  return {count:Number(counts[0][1]),rows};
}

async function tableScroller(tab) {
  const scroll = tab.playwright.getByRole('table',{name:'Virtual Table',exact:true})
    .locator('.bn-virtual-table');
  if (await readLocatorCount(scroll) !== 1) throw new Error('Working table scroller unavailable or ambiguous');
  return scroll;
}

async function scrollMetrics(scroll) {
  return readDOM(()=>scroll.evaluate(e => {
    const rect=e.getBoundingClientRect();
    return {left:rect.left,right:rect.right,top:rect.top,bottom:rect.bottom,
      scrollTop:e.scrollTop,clientHeight:e.clientHeight,scrollHeight:e.scrollHeight};
  },undefined,DOM_READ_OPTIONS));
}

function requireSupportedInput(tab) {
  for (const method of ['scroll','drag','click','getScreenshot','getAXState']) {
    if (typeof tab[method] !== 'function') throw new Error(`supported browser input unavailable: ${method}`);
  }
}

async function readDOM(operation) {
  // Retry only the same read. A native input is never repeated here.
  for (let attempt=0;attempt<3;attempt++) {
    try { return await operation(); }
    catch (error) { if (attempt===2) throw error; }
  }
}

async function scrollAt(tab, x, y, delta, viewport) {
  if (![x,y,delta,viewport.height].every(Number.isFinite) || delta===0 || viewport.height<=0) {
    throw new Error('invalid browser scroll geometry');
  }
  await tab.getScreenshot({emit:false});
  // The supported tab API takes a direction and viewport-page count, not pixels.
  // Callers remeasure the actual movement; the requested distance is not evidence.
  try {
    await tab.scroll([Math.round(x),Math.round(y)],delta>0?'down':'up',
      Math.max(0.1,Math.abs(delta)/viewport.height));
  } finally {
    // Virtualized fixed cells can still have a stale hit test at dispatch
    // completion. Observe the rendered result before deciding to wheel again.
    // Every caller then reads fresh table metrics or the exact details control.
    await tab.getScreenshot({emit:false});
  }
}

function requirePointerAnchor(anchor) {
  if (!anchor || !Number.isFinite(anchor.x) || !Number.isFinite(anchor.y)) {
    throw new Error('non-interactive pointer origin unavailable');
  }
  return anchor;
}

async function scrollTable(tab, metrics, delta) {
  const viewport=await tab.playwright.evaluate(() => ({height:innerHeight,width:innerWidth}),undefined,DOM_READ_OPTIONS);
  if (metrics.bottom<=80 || metrics.top>=viewport.height-40 || metrics.clientHeight<=0) {
    throw new Error('Working table scroller is outside viewport');
  }
  await scrollAt(tab,Math.max(20,Math.min(viewport.width-20,(metrics.left+metrics.right)/2)),
    Math.max(90,Math.min(viewport.height-50,(metrics.top+metrics.bottom)/2)),delta,viewport);
}

function matchRosterRow(actual,expected) {
  return actual.symbol===expected.symbol && actual.created===expected.created
    && actual.status===expected.status;
}

async function completeRoster(tab) {
  const initial=await rosterView(tab);
  if (initial.rows.length===initial.count) return validateRoster(initial.count,initial.rows);
  const scroll=await tableScroller(tab);
  let metrics=await scrollMetrics(scroll);
  if (metrics.scrollHeight-metrics.clientHeight<=2) {
    throw new Error(`PARTIAL roster: ${initial.rows.length}/${initial.count}`);
  }
  if (metrics.scrollTop>2) {
    await scrollTable(tab,metrics,-Math.ceil(metrics.scrollHeight+metrics.clientHeight));
    metrics=await scrollMetrics(scroll);
    if (metrics.scrollTop>2) throw new Error('Working table could not reach top');
  }
  const rows=[];
  const seen=new Map();
  let previous=null;
  let reachedBottom=false;
  for (let step=0;step<initial.count*4+8;step++) {
    const view=await rosterView(tab);
    if (view.count!==initial.count) throw new Error('UM Grid count changed during roster sweep');
    if (previous && !view.rows.some(row=>previous.some(old=>matchRosterRow(row,old)))) {
      throw new Error('Working table sweep skipped an unobserved interval');
    }
    for (const row of view.rows) {
      const old=seen.get(row.symbol);
      if (old && !matchRosterRow(row,old)) throw new Error('Working identity changed during roster sweep');
      if (!old) { seen.set(row.symbol,row); rows.push(row); }
    }
    metrics=await scrollMetrics(scroll);
    if (metrics.scrollTop>=metrics.scrollHeight-metrics.clientHeight-2) {
      reachedBottom=true;
      break;
    }
    previous=view.rows;
    const before=metrics.scrollTop;
    await scrollTable(tab,metrics,Math.max(60,Math.floor(metrics.clientHeight/2)));
    metrics=await scrollMetrics(scroll);
    if (metrics.scrollTop<=before+1) throw new Error('Working table sweep did not advance');
  }
  if (!reachedBottom) throw new Error('Working table sweep exceeded bound');
  return validateRoster(initial.count,rows);
}

async function focusSelectedGridTab(tab) {
  const view = await rosterView(tab);
  // Clicking the already-selected tab brings the table below the chart into view.
  // Re-read the complete roster afterward; this click grants no roster authority.
  try {
    await tab.playwright.getByRole('tab',{name:`UM Grid (${view.count})`,exact:true})
      .click({timeoutMs:DOM_READ_OPTIONS.timeoutMs});
  } catch (error) {
    // The tab was already selected. Resolve an uncertain navigation response
    // through fresh selected-tab/count proof, never by repeating native input.
    // Complete roster and visibility checks still follow this navigation step.
    const observed=await rosterView(tab);
    if (observed.count!==view.count) throw error;
  }
}

async function revealRosterRow(tab,target,roster) {
  const index=roster.findIndex(row=>matchRosterRow(row,target));
  if (index<0) throw new Error('drawer target absent from proven roster');
  const bySymbol=new Map(roster.map(row=>[row.symbol,row]));
  const initial=await rosterView(tab);
  if (initial.count!==roster.length) throw new Error('UM Grid count changed before drawer');
  for (const row of initial.rows) {
    const expected=bySymbol.get(row.symbol);
    if (!expected || !matchRosterRow(row,expected)) {
      throw new Error('Working identity changed before drawer');
    }
  }
  if (initial.rows.some(row=>matchRosterRow(row,target))) return;
  const scroll=await tableScroller(tab);
  for (let step=0;step<roster.length*4+8;step++) {
    const view=await rosterView(tab);
    if (view.count!==roster.length) throw new Error('UM Grid count changed before drawer');
    for (const row of view.rows) {
      const expected=bySymbol.get(row.symbol);
      if (!expected || !matchRosterRow(row,expected)) {
        throw new Error('Working identity changed before drawer');
      }
    }
    if (view.rows.some(row=>matchRosterRow(row,target))) return;
    const first=roster.findIndex(row=>row.symbol===view.rows[0].symbol);
    const last=roster.findIndex(row=>row.symbol===view.rows.at(-1).symbol);
    if (first<0 || last<first || (index>=first && index<=last)) {
      throw new Error('Working table order changed before drawer');
    }
    const metrics=await scrollMetrics(scroll);
    const before=metrics.scrollTop;
    const direction=index<first?-1:1;
    await scrollTable(tab,metrics,direction*Math.max(60,Math.floor(metrics.clientHeight/2)));
    const after=(await scrollMetrics(scroll)).scrollTop;
    if (Math.abs(after-before)<=1) throw new Error('drawer target cannot be revealed');
  }
  throw new Error('drawer target reveal exceeded bound');
}

export async function closeDrawer(tab) {
  const drawer = tab.playwright.getByRole('dialog', {name:'drawer', exact:true});
  if (await readLocatorCount(drawer) === 0) return;
  // Match the observed X icon; never click an anonymous button by position.
  const close = drawer.locator(`svg:has(path[d^="${CLOSE_PATH}"])`);
  for (let attempt = 0; attempt < 2; attempt++) {
    if (attempt > 0 && (await readLocatorCount(drawer) === 0 || !await drawer.isVisible())) return;
    if (await readLocatorCount(close) !== 1) throw new Error('drawer close control is ambiguous');
    let clickError = null;
    try { await close.click({timeoutMs:10000}); }
    catch (error) { clickError = error; }
    try { await drawer.waitFor({state:'hidden',timeoutMs:5000}); return; }
    catch (error) {
      if (await readLocatorCount(drawer) === 0 || !await drawer.isVisible()) return;
      if (clickError) throw clickError;
      if (attempt === 1) throw error;
    }
  }
}

async function readLocatorCount(locator) {
  for (let attempt=0;attempt<3;attempt++) {
    try { return await locator.count(); }
    catch (error) { if (attempt===2) throw error; }
  }
}

async function waitForVisible(locator, timeoutMs) {
  for (let attempt=0; attempt<3; attempt++) {
    try { await locator.waitFor({state:'visible',timeoutMs}); return; }
    catch (error) {
      // Bounded read-only waits tolerate the observed delayed rendering.
      // Confirm the same target; never repeat the click that opened it.
      if (await readLocatorCount(locator) === 1 && await locator.isVisible()) return;
      if (attempt === 2) throw error;
    }
  }
}

async function openDrawer(tab, target) {
  const symbol=target.symbol;
  const row = tab.playwright.getByRole('table', {name:'Virtual Table', exact:true})
    .getByRole('row').filter({hasText:new RegExp(`^\\s*${symbol}\\s+Perp`)});
  // The document icon was observed on this page; its tooltip is still required
  // on every capture. Never probe by clicking the End or modification buttons.
  const details = row.getByRole('cell').last().locator('button:has(path[d^="M19.1 5A1.1"])');
  {
    const button = details;
    let rect;
    const measure = async () => {
      const state=await readDOM(()=>button.evaluate((e,symbol) => {
      const r=e.getBoundingClientRect();
      const table=e.closest('.bn-virtual-table');
      const t=table?.getBoundingClientRect();
      const x=r.x+r.width/2, y=r.y+r.height/2;
      const hit=document.elementFromPoint(x,y);
      const tableRoot=e.closest('[role="table"],table');
      const sameRows=[...(tableRoot?.querySelectorAll('[role="row"],tr')||[])].filter(row=>{
        const cell=row.querySelector('[role="cell"],td');
        return new RegExp(`^\\s*${symbol}\\s+Perp`).test(cell?.textContent||'');
      });
      const cells=[...(e.closest('[role="row"],tr')?.querySelectorAll('[role="cell"],td')||[])];
      const actions=cells.at(-1);
      const forbidden='a,button,input,select,textarea,[role="button"],[role="link"],'
        +'[role="tab"],[role="table"],[role="dialog"],[contenteditable],canvas,iframe,[draggable="true"]';
      let anchor=null;
      for (const fraction of [0.5,1/3,2/3]) {
        const point={x:Math.round(innerWidth*fraction),y:30};
        const element=document.elementFromPoint(point.x,point.y);
        if (!element || element.closest(forbidden)) continue;
        if (!['DIV','HEADER','NAV','MAIN','BODY'].includes(element.tagName)) continue;
        if (!['auto','default'].includes(getComputedStyle(element).cursor)) continue;
        const selection=window.getSelection();
        anchor={...point,selectionCollapsed:selection===null || selection.isCollapsed};break;
      }
      return {x,y,hit:e===hit||e.contains(hit),anchor,
        rowCount:sameRows.length,buttonCount:actions?.querySelectorAll('button,[role="button"]').length||0,
        detailsCount:actions?.querySelectorAll('button:has(path[d^="M19.1 5A1.1"])').length||0,
        viewport:{height:innerHeight,width:innerWidth},
        describedBy:e.closest('[aria-describedby]')?.getAttribute('aria-describedby') || '',
        table:t?{left:t.left,right:t.right,top:t.top,bottom:t.bottom}:null};
      },symbol,DOM_READ_OPTIONS));
      if (state.rowCount!==1) throw new Error('active row disappeared or duplicated');
      if (!Number.isSafeInteger(state.buttonCount) || state.buttonCount<1 || state.buttonCount>8) {
        throw new Error('unrecognized action controls');
      }
      if (state.detailsCount!==1) throw new Error('View Details icon changed or is ambiguous');
      return state;
    };
    rect = await measure();
    const viewport=rect.viewport;
    for (let attempt=0; attempt<4; attempt++) {
      if (!rect.table) throw new Error('Working table scroll container unavailable');
      const table=rect.table;
      if (rect.hit && rect.y>=80 && rect.y<=viewport.height-40) break;
      try {
        const visibleTop=Math.max(80,table.top);
        const visibleBottom=Math.min(viewport.height-40,table.bottom);
        if (visibleBottom-visibleTop<80) {
          // Wheel over the page above the table, away from the chart and action controls.
          await scrollAt(tab,(table.left+table.right)/2,
            Math.max(100,Math.min(viewport.height-100,table.top-100)),table.top-150,viewport);
        } else {
          // A virtualized row can exist outside the table's visible area. Use
          // that area's center even when the outer page has reached its limit.
          // Table overflow alone does not prove that the outer page can move.
          const center=(visibleTop+visibleBottom)/2;
          if (rect.y===center) {
            await tab.getScreenshot({emit:false});
          } else {
            await scrollAt(tab,Math.max(table.left+20,Math.min(table.right-20,rect.x)),
              center,rect.y-center,viewport);
          }
        }
      } catch (error) {
        // The wheel command may have moved the page before losing its response.
        // Observe the same exact control; never issue an unverified second scroll.
        const observed=await measure();
        if (!observed.hit || observed.y<80 || observed.y>viewport.height-40) throw error;
      }
      // Also observe the last permitted correction. Never decide from the
      // pre-scroll rectangle or treat dispatch alone as proof of visibility.
      rect = await measure();
    }
    if (!rect?.hit || rect.y<80 || rect.y>viewport.height-40) {
      throw new Error('View Details control remains obscured or outside viewport');
    }
    const tooltip=tab.playwright.getByRole('tooltip',{name:'View Details',exact:true});
    for (let pointerAttempt=0;pointerAttempt<2;pointerAttempt++) {
    const anchor=requirePointerAnchor(rect.anchor);
    await tab.getScreenshot({emit:false});
    // A previous drag selects page text, including this blank origin. Starting
    // inside that selection suppresses the next tooltip in the observed UI.
    // Clear it with native input at the verified non-interactive point only.
    await tab.click([anchor.x,anchor.y]);
    // This fresh atomic read checks both the reset selection and the same icon.
    const afterReset=await measure();
    const cleared=requirePointerAnchor(afterReset.anchor);
    if (cleared.x!==anchor.x || cleared.y!==anchor.y || cleared.selectionCollapsed!==true) {
      throw new Error('pointer origin changed or selection did not clear; no details control clicked');
    }
    if (!afterReset.hit || Math.abs(afterReset.x-rect.x)>1 || Math.abs(afterReset.y-rect.y)>1) {
      throw new Error('View Details control moved or became obscured after pointer reset');
    }
    await tab.getScreenshot({emit:false});
    let dragError=null;
    // The supported API has drag, not hover. Start on verified non-interactive
    // chrome and release on the document icon; do not activate that control.
    // This selects page text; clear it again before the next strategy's drag.
    try { await tab.drag([anchor.x,anchor.y],[Math.round(rect.x),Math.round(rect.y)]); }
    catch (error) { dragError=error; }
    try {
      await waitForVisible(tooltip,DOM_READ_OPTIONS.timeoutMs);
      break;
    } catch (error) {
      if (dragError) throw dragError;
      if (pointerAttempt===1) throw new Error('View Details tooltip unavailable; no control clicked',{cause:error});
      // Only an absent tooltip after a completed pointer command permits one
      // new gesture. Re-prove the same control before resetting the safe origin;
      // uncertainty, movement and wrong ownership never authorize a click.
      const retryState=await measure();
      if (!retryState.hit || Math.abs(retryState.x-rect.x)>1 || Math.abs(retryState.y-rect.y)>1) {
        throw new Error('View Details control moved or became obscured');
      }
      rect=retryState;
    }
    }
    const description=await readDOM(()=>tooltip.evaluateAll(elements=>{
      if (elements.length!==1) return {count:elements.length,visible:false};
      const e=elements[0],r=e.getBoundingClientRect(),style=getComputedStyle(e);
      return {count:1,visible:r.width>0&&r.height>0&&style.visibility==='visible'&&style.display!=='none',
        id:e.id,text:e.innerText.trim()};
    },undefined,DOM_READ_OPTIONS));
    if (description.count!==1 || !description.visible) {
      throw new Error('View Details tooltip ambiguous or hidden; no control clicked');
    }
    const observed=await measure();
    if (!description.id || description.text!=='View Details'
      || !observed.describedBy.split(/\s+/).includes(description.id)) {
      throw new Error('View Details tooltip belongs to another control');
    }
    if (!observed.hit || Math.abs(observed.x-rect.x)>1 || Math.abs(observed.y-rect.y)>1) {
      throw new Error('View Details control moved or became obscured');
    }
    const current=await rosterView(tab);
    if (!current.rows.some(row=>matchRosterRow(row,target))) {
      throw new Error('Working identity changed before details click');
    }
    const drawer = tab.playwright.getByRole('dialog',{name:'drawer',exact:true});
    await tab.getScreenshot({emit:false});
    try { await tab.click([Math.round(rect.x),Math.round(rect.y)]); }
    catch (error) {
      // A transport timeout can occur after dispatch. Resolve by observing
      // the drawer, not by issuing the potentially duplicated click again.
      try { await waitForVisible(drawer,10000); } catch { throw error; }
    }
    await waitForVisible(drawer,10000);
    await waitForVisible(drawer.getByText('Strategy Number',{exact:true}),10000);
    return drawer;
  }
}

async function canonicalPath(value) {
  try { return await realpath(value); }
  catch (error) {
    if (platform()!=='win32' || error.code!=='EPERM') throw error;
    // The live Windows browser runtime rejected async realpath on this checkout,
    // while native realpath resolved it. Still require OS-backed canonicalization;
    // lexical path.resolve would fail to detect a junction escape.
    return realpathSync.native(value);
  }
}

function failureCauses(error) {
  const causes=[];
  const seen=new Set();
  for (let current=error; current && causes.length<5 && !seen.has(current); current=current.cause) {
    seen.add(current);
    const message=String(current.message || current);
    const classification=/before command dispatch/.test(message)?'browser_dispatch_timeout'
      :/Timed out|timed out/i.test(message)?'browser_wait_or_read_timeout'
      :/tooltip unavailable/.test(message)?'required_tooltip_unavailable':'unclassified';
    causes.push({name:current.name || 'Error',classification,
      message_sha256:createHash('sha256').update(message,'utf8').digest('hex')});
  }
  return causes;
}

async function containedDirectory(root, target) {
  const base = await canonicalPath(root);
  const relative = path.relative(base,target);
  if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('capture path escaped workspace');
  let current = base;
  for (const part of relative.split(path.sep).filter(Boolean)) {
    current = path.join(current,part);
    try { await mkdir(current); } catch (error) { if (error.code !== 'EEXIST') throw error; }
    const resolved = path.relative(base,await canonicalPath(current));
    if (resolved.startsWith('..') || path.isAbsolute(resolved)) throw new Error('capture path escaped workspace');
  }
}

export function selectInvestmentRoster(rows, amounts) {
  if (!Array.isArray(amounts) || !amounts.length || amounts.some(x=>!Number.isFinite(x)||x<=0)
    || new Set(amounts).size!==amounts.length) throw new Error('invalid investment filter');
  return rows.filter(row=>{
    const match=/^([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s+USDT(?:\s*\((?:Isolated|Cross)\))?$/.exec(row.investment_text||'');
    if (!match) throw new Error('unproven Invested Margin USDT field');
    return amounts.includes(Number(match[1].replaceAll(',','')));
  });
}

export function selectSymbolRoster(rows, symbols) {
  if (!Array.isArray(symbols) || !symbols.length || symbols.some(s=>!(/^[A-Z0-9]+USDT$/).test(s))
    || new Set(symbols).size!==symbols.length) throw new Error('invalid requested symbol scope');
  if (symbols.some(s=>!rows.some(row=>row.symbol===s))) throw new Error('requested symbol absent from complete Working roster');
  return rows.filter(row=>symbols.includes(row.symbol));
}

export async function captureDrawers(tab, {workspaceRoot, investmentAmounts=null, symbols=null}) {
  if (investmentAmounts!==null && symbols!==null) throw new Error('ambiguous capture scope');
  const root = await canonicalPath(workspaceRoot);
  const url = new URL(await tab.url());
  if (url.protocol !== 'https:' || !/^(www\.)?binance\.(com|bh)$/.test(url.hostname)
    || !url.pathname.includes('/trading-bots/futures/grid/')) throw new Error('not the verified Binance grid page');
  const started = new Date();
  const date = new Intl.DateTimeFormat('en-CA',{timeZone:'America/Lima',year:'numeric',month:'2-digit',day:'2-digit'}).format(started);
  const runId = `recurring_${started.toISOString().replace(/[^0-9]/g,'')}_${randomUUID().slice(0,8)}`;
  const staging = path.join(root,'outputs','runtime','brave_plugin_capture',runId);
  await containedDirectory(root,staging);
  const captures = [];
  let activeSymbol = null;
  try {
    requireSupportedInput(tab);
    await closeDrawer(tab);
    await focusSelectedGridTab(tab);
    const fullBefore = await completeRoster(tab);
    const before = symbols!==null?selectSymbolRoster(fullBefore,symbols)
      :investmentAmounts===null?fullBefore:selectInvestmentRoster(fullBefore,investmentAmounts);
    if (!before.length) throw new Error('no Working strategies match investment filter');
    for (const row of before) {
      activeSymbol = row.symbol;
      if (Date.now()-started.getTime() > CAPTURE_DEADLINE_MS) throw new Error('capture deadline exceeded');
      await revealRosterRow(tab,row,fullBefore);
      const drawer = await openDrawer(tab,row);
      try {
        const text = await readDOM(()=>drawer.innerText({timeoutMs:10000}));
        const capturedAt = new Date().toISOString();
        const rawDir = path.join(root,'Live',date,row.symbol,'drawer_captures',runId);
        await containedDirectory(root,rawDir);
        const rawPath = path.join(rawDir,'drawer.txt');
        await writeFile(rawPath,text,{encoding:'utf8',flag:'wx'});
        const identity = parseDrawer(text,row.symbol,row.created);
        if (investmentAmounts!==null) {
          const margin=/Invested Margin \(USDT\)\s+([0-9,.]+)/.exec(text);
          if (!margin || !investmentAmounts.includes(Number(margin[1].replaceAll(',','')))) {
            throw new Error('drawer investment does not match requested scope');
          }
        }
        captures.push({symbol:row.symbol,strategy_id:identity.strategyId,working_status:'Working',
          deployment_time_lima:identity.createdAt,captured_at_utc:capturedAt,raw_text_path:rawPath,
          raw_text_sha256:createHash('sha256').update(text,'utf8').digest('hex'),capture_status:'complete'});
      } finally {
        await closeDrawer(tab);
      }
    }
    const fullAfter = await completeRoster(tab);
    if (JSON.stringify(fullBefore) !== JSON.stringify(fullAfter)) throw new Error('Working roster changed during capture');
    const after=symbols!==null?selectSymbolRoster(fullAfter,symbols)
      :investmentAmounts===null?fullAfter:selectInvestmentRoster(fullAfter,investmentAmounts);
    if (Date.now()-started.getTime() > CAPTURE_DEADLINE_MS) throw new Error('capture deadline exceeded');
    const payload = {schema_version:'neutralgrid_brave_plugin_capture_bundle_v1',status:'complete',
      source:'brave_plugin',run_id:runId,page_identity:'Binance USD-M Futures Grid',source_url:url.href,
      authenticated:true,cycle_started_at_utc:started.toISOString(),cycle_completed_at_utc:new Date().toISOString(),
      working_row_count:before.length,roster_before_symbols:before.map(r=>r.symbol),
      roster_after_symbols:after.map(r=>r.symbol),roster_before_rows:before,roster_after_rows:after,captures,
      ...(investmentAmounts===null && symbols===null?{}:{capture_scope:{
        ...(symbols!==null?{type:'symbols',symbols}:{type:'invested_margin_usdt',amounts:investmentAmounts}),
        full_working_count:fullBefore.length,full_roster_before_rows:fullBefore,full_roster_after_rows:fullAfter}})};
    const bundlePath = path.join(staging,'capture_bundle.json');
    await writeFile(bundlePath,JSON.stringify(payload,null,2)+'\n',{flag:'wx'});
    return {status:'complete',bundlePath,runId,symbols:before.map(r=>r.symbol)};
  } catch (error) {
    let cleanupError = null;
    try { await closeDrawer(tab); } catch (cleanup) { cleanupError = String(cleanup); }
    await writeFile(path.join(staging,'failure.json'),JSON.stringify({status:'blocked',run_id:runId,
      started_at_utc:started.toISOString(),failed_at_utc:new Date().toISOString(),error:String(error),
      active_symbol:activeSymbol,completed_capture_count:captures.length,cleanup_error:cleanupError,
      failure_causes:failureCauses(error)},null,2),{flag:'wx'});
    throw error;
  }
}
