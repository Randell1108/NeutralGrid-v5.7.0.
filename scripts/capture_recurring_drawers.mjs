// Run only inside the Chrome skill's persistent JavaScript tool, with a claimed
// authenticated tab. Browser methods below are the documented browser-client API.
import { mkdir, writeFile, realpath } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import path from 'node:path';

const CLOSE_PATH = 'M4.863 17.863L10.726 12 4.863 6.137';
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
  const tabs = await tab.playwright.getByRole('tab').allTextContents({timeoutMs:5000});
  const counts = tabs.map(t => /^UM Grid \((\d+)\)$/.exec(t.trim())).filter(Boolean);
  if (counts.length !== 1) throw new Error('authenticated UM Grid count unavailable');
  for (const name of [counts[0][0], 'Running']) {
    const selected = await tab.playwright.getByRole('tab',{name,exact:true})
      .evaluate(e => e.getAttribute('aria-selected'));
    if (selected !== 'true') throw new Error(`required tab is not selected: ${name}`);
  }
  const rows = await tab.playwright.getByRole('table', {name:'Virtual Table', exact:true})
    .getByRole('row').evaluateAll(elements => elements.flatMap(row => {
      const cells = [...row.querySelectorAll('[role="cell"]')];
      if (!cells.length) return [];
      const symbol = (cells[0].textContent || '').trim().match(/^([A-Z0-9]+USDT)\s+Perp/);
      const created = (cells[1]?.textContent || '').trim().replace(/(\d{4}-\d{2}-\d{2})\s*(\d{2}:\d{2}:\d{2})/, '$1 $2');
      return [{symbol:symbol?.[1] || '',created,status:(cells[14]?.textContent || '').trim()}];
    }));
  validateRoster(rows.length,rows);
  return {count:Number(counts[0][1]),rows};
}

async function tableScroller(tab) {
  const scroll = tab.playwright.getByRole('table',{name:'Virtual Table',exact:true})
    .locator('.bn-virtual-table');
  if (await scroll.count() !== 1) throw new Error('Working table scroller unavailable or ambiguous');
  return scroll;
}

async function scrollMetrics(scroll) {
  return scroll.evaluate(e => {
    const rect=e.getBoundingClientRect();
    return {left:rect.left,right:rect.right,top:rect.top,bottom:rect.bottom,
      scrollTop:e.scrollTop,clientHeight:e.clientHeight,scrollHeight:e.scrollHeight};
  });
}

async function scrollTable(tab, metrics, delta) {
  const viewport=await tab.playwright.evaluate(() => ({height:innerHeight,width:innerWidth}));
  if (metrics.bottom<=80 || metrics.top>=viewport.height-40 || metrics.clientHeight<=0) {
    throw new Error('Working table scroller is outside viewport');
  }
  await tab.cua.scroll({x:Math.round(Math.max(20,Math.min(viewport.width-20,
    (metrics.left+metrics.right)/2))),
  y:Math.round(Math.max(90,Math.min(viewport.height-50,(metrics.top+metrics.bottom)/2))),
  scrollY:delta,scrollX:0});
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
  await tab.playwright.getByRole('tab',{name:`UM Grid (${view.count})`,exact:true})
    .click({timeoutMs:5000});
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
  if (await readDrawerCount(drawer) === 0) return;
  // Match the observed X icon; never click an anonymous button by position.
  const close = drawer.locator(`svg:has(path[d^="${CLOSE_PATH}"])`);
  for (let attempt = 0; attempt < 2; attempt++) {
    if (await readDrawerCount(drawer) === 0 || (attempt > 0 && !await drawer.isVisible())) return;
    if (await close.count() !== 1) throw new Error('drawer close control is ambiguous');
    let clickError = null;
    try { await close.click({timeoutMs:10000}); }
    catch (error) { clickError = error; }
    try { await drawer.waitFor({state:'hidden',timeoutMs:5000}); return; }
    catch (error) {
      if (await readDrawerCount(drawer) === 0 || !await drawer.isVisible()) return;
      if (clickError) throw clickError;
      if (attempt === 1) throw error;
    }
  }
}

async function readDrawerCount(drawer) {
  for (let attempt=0;attempt<3;attempt++) {
    try { return await drawer.count(); }
    catch (error) { if (attempt===2) throw error; }
  }
}

async function waitForVisible(locator, timeoutMs) {
  for (let attempt=0; attempt<3; attempt++) {
    try { await locator.waitFor({state:'visible',timeoutMs}); return; }
    catch (error) {
      // Bounded read-only waits tolerate the observed delayed rendering.
      // Confirm the same target; never repeat the click that opened it.
      if (await locator.count() === 1 && await locator.isVisible()) return;
      if (attempt === 2) throw error;
    }
  }
}

async function openDrawer(tab, symbol) {
  const row = tab.playwright.getByRole('table', {name:'Virtual Table', exact:true})
    .getByRole('row').filter({hasText:`${symbol} Perp`});
  if (await row.count() !== 1) throw new Error('active row disappeared or duplicated');
  const buttons = row.getByRole('cell').last().getByRole('button');
  const total = await buttons.count();
  if (total < 1 || total > 8) throw new Error('unrecognized action controls');
  // The document icon was observed on this page; its tooltip is still required
  // on every capture. Never probe by clicking the End or modification buttons.
  const details = row.getByRole('cell').last().locator('button:has(path[d^="M19.1 5A1.1"])');
  if (await details.count() !== 1) throw new Error('View Details icon changed or is ambiguous');
  {
    const button = details;
    const viewport = await tab.playwright.evaluate(() => ({height:innerHeight,width:innerWidth}));
    let rect;
    const measure = () => button.evaluate(e => {
      const r=e.getBoundingClientRect();
      const table=e.closest('.bn-virtual-table');
      const t=table?.getBoundingClientRect();
      const x=r.x+r.width/2, y=r.y+r.height/2;
      const hit=document.elementFromPoint(x,y);
      return {x,y,hit:e===hit||e.contains(hit),
        table:t?{left:t.left,right:t.right,top:t.top,bottom:t.bottom}:null};
    });
    for (let attempt=0; attempt<4; attempt++) {
      rect = await measure();
      if (!rect.table) throw new Error('Working table scroll container unavailable');
      const table=rect.table;
      if (rect.hit && rect.y>=80 && rect.y<=viewport.height-40) break;
      try {
        if (table.top<80 || table.bottom>viewport.height-20) {
          // Wheel over the page above the table, away from the chart and action controls.
          await tab.cua.scroll({x:Math.round((table.left+table.right)/2),
            y:Math.max(100,Math.min(viewport.height-100,table.top-100)),
            scrollY:Math.round(table.top-150),scrollX:0});
        } else {
          // The table has its own scroller. A row in the DOM can be clipped by it.
          await tab.cua.scroll({x:Math.round(Math.max(table.left+20,Math.min(table.right-20,rect.x))),
            y:Math.round((table.top+Math.min(table.bottom,viewport.height-40))/2),
            scrollY:Math.round(rect.y-(table.top+56)),scrollX:0});
        }
      } catch (error) {
        // The wheel command may have moved the page before losing its response.
        // Observe the same exact control; never issue an unverified second scroll.
        const observed=await measure();
        if (!observed.hit || observed.y<80 || observed.y>viewport.height-40) throw error;
      }
    }
    if (!rect?.hit || rect.y<80 || rect.y>viewport.height-40) {
      throw new Error('View Details control remains obscured or outside viewport');
    }
    await tab.cua.move({x:viewport.width-16,y:40});
    await tab.cua.move({x:Math.round(rect.x),y:Math.round(rect.y)});
    try {
      await waitForVisible(tab.playwright.getByRole('tooltip',{name:'View Details',exact:true}),3000);
    } catch {
      throw new Error('View Details tooltip unavailable; no control clicked');
    }
    const drawer = tab.playwright.getByRole('dialog',{name:'drawer',exact:true});
    try { await tab.cua.click({x:Math.round(rect.x),y:Math.round(rect.y)}); }
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

async function containedDirectory(root, target) {
  const base = await realpath(root);
  const relative = path.relative(base,target);
  if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('capture path escaped workspace');
  let current = base;
  for (const part of relative.split(path.sep).filter(Boolean)) {
    current = path.join(current,part);
    try { await mkdir(current); } catch (error) { if (error.code !== 'EEXIST') throw error; }
    const resolved = path.relative(base,await realpath(current));
    if (resolved.startsWith('..') || path.isAbsolute(resolved)) throw new Error('capture path escaped workspace');
  }
}

export async function captureDrawers(tab, {workspaceRoot}) {
  const root = await realpath(workspaceRoot);
  const url = new URL(await tab.url());
  if (url.protocol !== 'https:' || !/^(www\.)?binance\.(com|bh)$/.test(url.hostname)
    || !url.pathname.includes('/trading-bots/futures/grid/')) throw new Error('not the verified Binance grid page');
  const started = new Date();
  const date = new Intl.DateTimeFormat('en-CA',{timeZone:'America/Lima',year:'numeric',month:'2-digit',day:'2-digit'}).format(started);
  const runId = `recurring_${started.toISOString().replace(/[^0-9]/g,'')}_${randomUUID().slice(0,8)}`;
  const staging = path.join(root,'outputs','runtime','chrome_plugin_capture',runId);
  await containedDirectory(root,staging);
  const captures = [];
  let activeSymbol = null;
  try {
    await closeDrawer(tab);
    await focusSelectedGridTab(tab);
    const before = await completeRoster(tab);
    for (const row of before) {
      activeSymbol = row.symbol;
      if (Date.now()-started.getTime() > 240000) throw new Error('capture deadline exceeded');
      await revealRosterRow(tab,row,before);
      const drawer = await openDrawer(tab,row.symbol);
      try {
        const text = await drawer.innerText({timeoutMs:10000});
        const capturedAt = new Date().toISOString();
        const rawDir = path.join(root,'Live',date,row.symbol,'drawer_captures',runId);
        await containedDirectory(root,rawDir);
        const rawPath = path.join(rawDir,'drawer.txt');
        await writeFile(rawPath,text,{encoding:'utf8',flag:'wx'});
        const identity = parseDrawer(text,row.symbol,row.created);
        captures.push({symbol:row.symbol,strategy_id:identity.strategyId,working_status:'Working',
          deployment_time_lima:identity.createdAt,captured_at_utc:capturedAt,raw_text_path:rawPath,
          raw_text_sha256:createHash('sha256').update(text,'utf8').digest('hex'),capture_status:'complete'});
      } finally {
        await closeDrawer(tab);
      }
    }
    const after = await completeRoster(tab);
    if (JSON.stringify(before) !== JSON.stringify(after)) throw new Error('Working roster changed during capture');
    const payload = {schema_version:'neutralgrid_chrome_plugin_capture_bundle_v1',status:'complete',
      source:'chrome_plugin',run_id:runId,page_identity:'Binance USD-M Futures Grid',source_url:url.href,
      authenticated:true,cycle_started_at_utc:started.toISOString(),cycle_completed_at_utc:new Date().toISOString(),
      working_row_count:before.length,roster_before_symbols:before.map(r=>r.symbol),
      roster_after_symbols:after.map(r=>r.symbol),roster_before_rows:before,roster_after_rows:after,captures};
    const bundlePath = path.join(staging,'capture_bundle.json');
    await writeFile(bundlePath,JSON.stringify(payload,null,2)+'\n',{flag:'wx'});
    return {status:'complete',bundlePath,runId,symbols:before.map(r=>r.symbol)};
  } catch (error) {
    let cleanupError = null;
    try { await closeDrawer(tab); } catch (cleanup) { cleanupError = String(cleanup); }
    await writeFile(path.join(staging,'failure.json'),JSON.stringify({status:'blocked',run_id:runId,
      started_at_utc:started.toISOString(),failed_at_utc:new Date().toISOString(),error:String(error),
      active_symbol:activeSymbol,completed_capture_count:captures.length,cleanup_error:cleanupError},null,2),{flag:'wx'});
    throw error;
  }
}
