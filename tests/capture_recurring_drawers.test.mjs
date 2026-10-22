import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, readdir, symlink } from 'node:fs/promises';
import { realpathSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { captureDrawers, parseDrawer, validateRoster, selectInvestmentRoster, selectSymbolRoster } from '../scripts/capture_recurring_drawers.mjs';

const created = '2026-09-25 09:23:39';
const valid = `BTCUSDT\nWorking\nTime Created\n${created}\nTotal Profit (USDT)\nInvested Margin (USDT)\nMatched Profit (USDT)\nPositions\nPending Order\nGrid Details\nOrder History\nStrategy Number\n123`;
test('drawer binds exact symbol, strategy, and deployment time', () => {
  assert.deepEqual(parseDrawer(valid,'BTCUSDT',created),{strategyId:'123',createdAt:created});
});
test('same symbol replacement is rejected by deployment fence', () => {
  assert.throws(()=>parseDrawer(valid,'BTCUSDT','2026-09-25 09:24:39'),/changed deployment/);
});
test('partial and ambiguous drawer cannot be marked complete', () => {
  assert.throws(()=>parseDrawer(valid.replace('Positions',''),'BTCUSDT',created),/incomplete/);
  assert.throws(()=>parseDrawer(valid+'\nStrategy Number\n124','BTCUSDT',created),/ambiguous/);
  assert.throws(()=>parseDrawer(valid,'ETHUSDT',created),/symbol mismatch/);
});
test('observed History format requires matched profit and trade totals',()=>{
  const current=valid.replace('Order History','History\nTotal Matched Profit\nTotal Matched Trades');
  assert.equal(parseDrawer(current,'BTCUSDT',created).strategyId,'123');
  assert.throws(()=>parseDrawer(valid.replace('Order History','History'),'BTCUSDT',created),/matched history/);
});
const row = {symbol:'BTCUSDT',created,status:'Working'};

test('investment filter uses exact displayed USDT margin, not leverage/notional',()=>{
  const rows=[{...row,investment_text:'1,000.00 USDT(Isolated)'},
    {...row,symbol:'ETHUSDT',investment_text:'400.00 USDT(Isolated)'},
    {...row,symbol:'SOLUSDT',investment_text:'40.00 USDT(Isolated)'}];
  assert.deepEqual(selectInvestmentRoster(rows,[1000,400]).map(x=>x.symbol),['BTCUSDT','ETHUSDT']);
  assert.throws(()=>selectInvestmentRoster([{...row,investment_text:'400 USD'}],[400]),/unproven/);
});

test('explicit symbol scope rejects absent or duplicate requested symbols',()=>{
  assert.throws(()=>selectSymbolRoster([row],['ETHUSDT']),/absent/);
  assert.throws(()=>selectSymbolRoster([row],['BTCUSDT','BTCUSDT']),/invalid/);
});
test('partial virtualized roster does not silently drop bots', () => {
  assert.throws(()=>validateRoster(2,[row]),/PARTIAL/);
});
test('duplicate symbols and non-Working rows are rejected', () => {
  assert.throws(()=>validateRoster(2,[row,row]),/duplicate/);
  assert.throws(()=>validateRoster(1,[{...row,status:'Expired'}]),/invalid/);
});
test('malformed symbol and missing count are rejected', () => {
  assert.throws(()=>validateRoster(0,[]),/no proven/);
  assert.throws(()=>validateRoster(1,[{...row,symbol:'../BTCUSDT'}]),/invalid/);
  assert.deepEqual(validateRoster(1,[row]),[row]);
});

function fakeTab(options={}) {
  const events = [];
  let opened = false, y = options.tableInitiallyOffscreen ? 1000 : options.fixedPage ? -80 : options.nearBottom ? -19.6 : options.occluded ? 329 : 800;
  let tableTop = options.tableInitiallyOffscreen ? 950 : options.fixedPage ? 264.8 : options.nearBottom ? 203.2 : options.occluded ? 616 : 100;
  let scrollCount = 0;
  let paintPending = false;
  let tooltipReads = 0, tooltipWaitBudget = 0, closeReads = 0;
  let detailReads = 0;
  let rosterReads = 0, discoveryReads = 0, closeClicks = 0, hoverCount = 0;
  let selectionActive=Boolean(options.selectionBlocksTooltip),tooltipBlocked=false,originClicks=0;
  const details = {
    count:async()=>1,
    evaluate:async()=>{
      events.push('detail-state-read');
      if(options.unavailableDetailState||(options.transientDetailRead&&detailReads++===0))throw new Error('details read dispatch timed out');
      return {x:options.movedOnHover&&hoverCount?150:100,y,
      rowCount:options.duplicateRow?2:1,buttonCount:options.unknownActions?9:4,detailsCount:options.duplicateDetails?2:1,
      viewport:{width:1200,height:700},
      anchor:options.noSafeAnchor?null:{x:options.anchorMovesAfterReset&&originClicks?1000:1100,y:40,selectionCollapsed:!selectionActive},
      describedBy:'details-tooltip',hit:!paintPending&&y>=tableTop&&y<=(options.nearBottom?634.2:tableTop+425)&&y<=660,
      table:{left:0,right:500,top:tableTop,bottom:options.nearBottom?634.2:tableTop+425}};
    },
  };
  const rowLocator = {
    count:async()=>1,
    getByRole:()=>({last:()=>({getByRole:()=>({count:async()=>4}),locator:()=>details})}),
  };
  const drawer = {
    count:async()=>{
      if (options.transientDrawerCount && discoveryReads++===0) throw new Error('drawer count dispatch timed out');
      return opened?(options.lateDiscovery && discoveryReads++===0?0:1):0;
    },
    isVisible:async()=>opened && !options.neverVisible,
    locator:()=>({count:async()=>{
      if(options.transientCloseCount&&closeReads++===0)throw new Error('close count dispatch timed out');
      return 1;
    },click:async()=>{
      closeClicks++;
      if (!options.firstCloseIgnored || closeClicks > 1) opened=false;
      events.push('close');
      if(options.closeResponseLost)throw new Error('close response lost');
    }}),
    waitFor:async({state})=>{
      if((state==='visible')!==opened || (state==='visible' && options.delayedVisibility))throw new Error('drawer wait timed out');
    },
    getByText:()=>({waitFor:async()=>{}}),
    innerText:async()=>options.invalidDrawer?'invalid':valid,
  };
  const tab = {
    url:async()=> 'https://www.binance.com/en/trading-bots/futures/grid/BTCUSDT',
    getScreenshot:async()=>{events.push('screenshot');paintPending=false;},
    getAXState:async()=>{events.push('state-read');},
    click:async(point)=>{
        if(point[0]===1100&&point[1]===40){
          originClicks++;events.push('origin-click');
          selectionActive=Boolean(options.selectionResetIgnored);
          if(options.originResponseLost)throw new Error('origin response lost');
          return;
        }
        events.push('details-click');opened=true;
        if(options.clickResponseLost)throw new Error('click response lost');
    },
    scroll:async ([x,pointY],direction,pages)=>{
      events.push({x,y:pointY,direction,pages,scrollY:direction==='down'?pages:-pages});
      scrollCount++;
      if(options.fixedPage){
        // The page is at its limit. Only a wheel inside the visible table moves its row.
        if(pointY>=tableTop&&pointY<=660)y=400;
      }else if(options.visibleOnFourthScroll){
        if(scrollCount===4)y=300;
      }else if(!options.stuckScroll && !options.tableInitiallyOffscreen){
        if(tableTop>560){tableTop=150;y=-137;}else y=300;
      }
      paintPending=Boolean(options.delayedScrollPaint);
      if(options.scrollResponseLost) throw new Error('scroll response lost');
    },
    drag:async()=>{
      tooltipBlocked=Boolean((options.selectionBlocksTooltip&&selectionActive)
        ||(options.firstTooltipMissing&&hoverCount===0));selectionActive=true;
      hoverCount++;events.push('details-hover');
      if(options.dragResponseLost)throw new Error('drag response lost');
    },
    playwright:{
      evaluate:async()=>({width:1200,height:700,
        selectionCollapsed:!selectionActive,
        ...(options.noSafeAnchor?{}:{x:options.anchorMovesAfterReset&&originClicks?1000:1100,y:40})}),
      getByRole:(role)=>{
        if(role==='dialog')return drawer;
        if(role==='tooltip')return {
          count:async()=>{
            if(options.transientTooltipCount&&tooltipReads++===0)throw new Error('tooltip count dispatch timed out');
            return options.noTooltip||tooltipBlocked?0:options.duplicateTooltip?2:1;
          },
          isVisible:async()=>!options.noTooltip&&!tooltipBlocked&&(!options.slowTooltipDispatch||tooltipWaitBudget>=5000),
          evaluate:async()=>({id:options.wrongTooltipBinding?'other-control':'details-tooltip',text:'View Details'}),
          evaluateAll:async()=>({count:options.noTooltip||tooltipBlocked?0:options.duplicateTooltip?2:1,
            visible:!options.noTooltip&&!tooltipBlocked,
            id:options.wrongTooltipBinding?'other-control':'details-tooltip',text:'View Details'}),
          waitFor:async({timeoutMs})=>{
            tooltipWaitBudget=timeoutMs;
            if(options.slowTooltipDispatch&&timeoutMs<5000)throw new Error('CDP operation exceeded its deadline before command dispatch');
            if(options.noTooltip||tooltipBlocked||(options.transientTooltipCount&&tooltipReads===0))throw new Error('no tooltip');events.push('tooltip');
          },
        };
        if(role==='tab')return {
          allTextContents:async()=>[`UM Grid (${options.partial?2:1})`],
          evaluateAll:async()=>{
            rosterReads++;
            return {tabs:[{text:`UM Grid (${options.partial?2:1})`,selected:options.wrongTab?'false':'true'},
              {text:'Running',selected:'true'}],tableCount:options.duplicateTable?2:1,
              rows:[{...row,created:(options.changedRoster&&rosterReads>4)
                ||(options.replacedOnHover&&hoverCount>0)?'2026-09-25 10:00:00':created}]};
          },
          evaluate:async()=>options.wrongTab?'false':'true',
          click:async()=>{events.push('focus-grid'); if(options.tableInitiallyOffscreen){tableTop=150;y=300;}
            if(options.gridFocusResponseLost)throw new Error('selected tab click response lost');},
        };
        if(role==='table')return {getByRole:()=>({
          filter:()=>rowLocator,
          evaluateAll:async()=>{
            rosterReads++;
            return [{...row,created:(options.changedRoster&&rosterReads>4)
              ||(options.replacedOnHover&&hoverCount>0)?'2026-09-25 10:00:00':created}];
          },
        })};
        throw new Error(`unexpected role ${role}`);
      },
    },
  };
  if(options.noNativeDrag)delete tab.drag;
  return {tab,events};
}

async function temporaryRoot(t) {
  const root = await mkdtemp(path.join(os.tmpdir(),'neutralgrid-drawer-test-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  return realpathSync.native(root);
}

test('capture requires tooltip before click, scrolls gutter, and stores raw only under Live', async t=>{
  const root = await temporaryRoot(t), {tab,events}=fakeTab();
  const result = await captureDrawers(tab,{workspaceRoot:root});
  const bundle = JSON.parse(await readFile(result.bundlePath,'utf8'));
  assert.equal(result.status,'complete');
  assert.equal(bundle.captures[0].strategy_id,'123');
  assert.ok(bundle.captures[0].raw_text_path.startsWith(path.join(root,'Live')+path.sep));
  assert.equal(await readFile(bundle.captures[0].raw_text_path,'utf8'),valid);
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
  assert.ok(events.some(e=>typeof e==='object'&&typeof e.scrollY==='number'));
  assert.deepEqual(await readdir(path.dirname(result.bundlePath)),['capture_bundle.json']);
});
test('offscreen grid is focused before opening a drawer',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({tableInitiallyOffscreen:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.ok(events.indexOf('focus-grid')<events.indexOf('details-click'));
});

test('uncertain already-selected tab navigation is observed without a second click',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({tableInitiallyOffscreen:true,gridFocusResponseLost:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='focus-grid').length,1);
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('bounded DOM dispatch budget tolerates a slow read without skipping selected-tab identity',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab();
  const original=tab.playwright.getByRole;
  const budgets=[];
  tab.playwright.getByRole=(role,...args)=>{
    const locator=original(role,...args);
    if(role==='tab')locator.evaluateAll=async(_fn,_arg,options)=>{
      const budget=options?.timeoutMs ?? 3000;
      budgets.push(budget);
      if(budget<5000)throw new Error('CDP operation exceeded its deadline before command dispatch');
      return {tabs:[{text:'UM Grid (1)',selected:'true'},{text:'Running',selected:'true'}],tableCount:1,rows:[row]};
    };
    return locator;
  };
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.ok(budgets.length>0 && budgets.every(budget=>budget===10000));
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('scoped DOM proofs preserve fresh state without redundant full-page exports',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab();
  tab.playwright.domSnapshot=async()=>{throw new Error('redundant full DOM export');};
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.ok(!events.includes('state-read'));
  assert.ok(events.slice(events.indexOf('origin-click')+1,events.indexOf('details-hover')).includes('detail-state-read'));
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('atomic roster proof rejects ambiguous tables and uses no second row dispatch',async t=>{
  const root=await temporaryRoot(t),{tab}=fakeTab({duplicateTable:true});
  await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/table unavailable or ambiguous/);
  const good=fakeTab();
  const original=good.tab.playwright.getByRole;
  good.tab.playwright.getByRole=(role,...args)=>{
    const locator=original(role,...args);
    if(role==='table'){
      const nested=locator.getByRole;
      locator.getByRole=(...nestedArgs)=>{
        const rows=nested(...nestedArgs);
        rows.evaluateAll=async()=>{throw new Error('redundant row dispatch');};
        return rows;
      };
    }
    return locator;
  };
  assert.equal((await captureDrawers(good.tab,{workspaceRoot:root})).status,'complete');
});

test('unavailable scoped DOM proof cannot authorize another UI action',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({unavailableDetailState:true});
  await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/details read dispatch timed out/);
  assert.ok(!events.includes('details-click'));
});

test('transient scoped DOM read retries without repeating native input',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({transientDetailRead:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-click').length,1);
  assert.equal(events.filter(e=>e==='details-hover').length,1);
});

for (const options of [{wrongTab:true},{wrongTooltipBinding:true},{duplicateTooltip:true},
  {replacedOnHover:true},{movedOnHover:true}]) {
  test(`fresh DOM path retains identity rejection: ${JSON.stringify(options)}`,async t=>{
    const root=await temporaryRoot(t),{tab,events}=fakeTab(options);
    tab.playwright.domSnapshot=async()=> 'fresh rendered DOM';
    await assert.rejects(captureDrawers(tab,{workspaceRoot:root}));
    assert.ok(!events.includes('details-click'));
  });
}

test('Live junction escape is rejected before raw data can leave the workspace',async t=>{
  const root=await temporaryRoot(t),outside=await temporaryRoot(t),{tab}=fakeTab();
  await symlink(outside,path.join(root,'Live'),process.platform==='win32'?'junction':'dir');
  await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/path escaped workspace/);
  assert.deepEqual(await readdir(outside),[]);
});

test('current supported tab API captures without a legacy cua object',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab();
  assert.equal('cua' in tab,false);
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.ok(events.indexOf('screenshot')<events.indexOf('details-hover'));
  assert.ok(events.indexOf('details-hover')<events.indexOf('tooltip'));
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('lost drag response is accepted only through the same bound visible tooltip',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({dragResponseLost:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-hover').length,1);
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('selected pointer origin is cleared and verified before tooltip movement',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({selectionBlocksTooltip:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.ok(events.indexOf('origin-click')<events.indexOf('details-hover'));
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('tooltip ownership failure retains its specific diagnostic',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({wrongTooltipBinding:true});
  await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/tooltip belongs to another control/);
  assert.ok(!events.includes('details-click'));
});
test('transient read-only drawer check is retried before capture',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({transientDrawerCount:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('transient tooltip count retries the same locator before any details click',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({transientTooltipCount:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-hover').length,1);
  assert.equal(events.filter(e=>e==='details-click').length,1);
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('tooltip dispatch receives the bounded DOM budget without repeating native movement',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({slowTooltipDispatch:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-hover').length,1);
  assert.equal(events.filter(e=>e==='details-click').length,1);
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('transient close-control count retries only the read before one close click',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({transientCloseCount:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='close').length,1);
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('failed tooltip records bounded hashed cause metadata without clicking details',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({noTooltip:true});
  await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/tooltip unavailable/);
  const runs=await readdir(path.join(root,'outputs','runtime','brave_plugin_capture'));
  const failure=JSON.parse(await readFile(path.join(root,'outputs','runtime','brave_plugin_capture',runs[0],'failure.json'),'utf8'));
  assert.equal(failure.failure_causes[0].classification,'required_tooltip_unavailable');
  assert.equal(failure.failure_causes.length,2);
  assert.ok(failure.failure_causes.every(cause=>/^[a-f0-9]{64}$/.test(cause.message_sha256)));
  assert.ok(failure.failure_causes.every(cause=>!('message' in cause)));
  assert.ok(!events.includes('details-click'));
});

test('one absent tooltip permits a fresh proven pointer reset before one details click',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({firstTooltipMissing:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-hover').length,2);
  assert.equal(events.filter(e=>e==='origin-click').length,2);
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('permanently absent tooltip exhausts two gestures without clicking details',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({noTooltip:true});
  await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/tooltip unavailable/);
  assert.equal(events.filter(e=>e==='details-hover').length,2);
  assert.equal(events.filter(e=>e==='details-click').length,0);
});
test('lost scroll response is accepted only after the target becomes hit-testable',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({scrollResponseLost:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>typeof e==='object'&&typeof e.scrollY==='number').length,1);
  assert.equal(events.filter(e=>e==='details-click').length,1);
});
test('lost scroll response with no visible movement still blocks capture',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({scrollResponseLost:true,stuckScroll:true});
  await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/scroll response lost/);
  assert.ok(!events.includes('details-click'));
});
test('obscured row scrolls page and inner table before tooltip and click',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({occluded:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>typeof e==='object').length,2);
  assert.equal(events.filter(e=>typeof e==='object')[1].y,363);
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});
test('fractional visible table edge uses its inner scroller',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({nearBottom:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>typeof e==='object').length,1);
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('clipped row uses the visible inner table when the outer page cannot move',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({fixedPage:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  const wheels=events.filter(e=>typeof e==='object');
  assert.equal(wheels.length,1);
  assert.ok(wheels[0].y>=264.8&&wheels[0].y<=660);
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('fourth scroll is remeasured before rejecting an obscured control',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({visibleOnFourthScroll:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>typeof e==='object').length,4);
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('post-scroll paint is observed before another wheel can chase a stale hit test',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({delayedScrollPaint:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>typeof e==='object').length,1);
  assert.ok(events.indexOf('tooltip')<events.indexOf('details-click'));
});

test('late visible drawer is confirmed without a second click',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({delayedVisibility:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('uncertain click completion is resolved by observation, not another click',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({clickResponseLost:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('visible drawer after ignored close is rechecked before a bounded retry',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({firstCloseIgnored:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='close').length,2);
});
test('uncertain close completion is resolved by observing the hidden drawer',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({closeResponseLost:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='close').length,1);
});

test('drawer appearing on a later bounded wait does not trigger a second click',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({delayedVisibility:true,lateDiscovery:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-click').length,1);
});

test('a drawer still invisible after timeout is rejected and closed',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({delayedVisibility:true,neverVisible:true});
  await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/drawer wait timed out/);
  assert.ok(events.includes('close'));
});

for (const options of [{noTooltip:true},{stuckScroll:true},{partial:true},{wrongTab:true},
  {duplicateRow:true},{unknownActions:true},{duplicateDetails:true},
  {wrongTooltipBinding:true},{duplicateTooltip:true},{noSafeAnchor:true},{noNativeDrag:true},
  {replacedOnHover:true},{movedOnHover:true},{dragResponseLost:true,noTooltip:true}]) {
  test(`unsafe acquisition stops before any drawer click: ${JSON.stringify(options)}`,async t=>{
    const root=await temporaryRoot(t),{tab,events}=fakeTab(options);
    await assert.rejects(captureDrawers(tab,{workspaceRoot:root}));
    assert.ok(!events.includes('details-click'));
    const staging=path.join(root,'outputs','runtime','brave_plugin_capture');
    const runs=await readdir(staging);
    assert.deepEqual(await readdir(path.join(staging,runs[0])),['failure.json']);
  });
}

for(const options of [{selectionResetIgnored:true},{anchorMovesAfterReset:true},{originResponseLost:true}]){
  test(`unverified pointer reset stops before hover or drawer click: ${JSON.stringify(options)}`,async t=>{
    const root=await temporaryRoot(t),{tab,events}=fakeTab(options);
    await assert.rejects(captureDrawers(tab,{workspaceRoot:root}));
    assert.ok(!events.includes('details-hover'));
    assert.ok(!events.includes('details-click'));
  });
}

for (const options of [{changedRoster:true},{invalidDrawer:true}]) {
  test(`incomplete capture closes drawer and publishes failure only: ${JSON.stringify(options)}`,async t=>{
    const root=await temporaryRoot(t),{tab,events}=fakeTab(options);
    await assert.rejects(captureDrawers(tab,{workspaceRoot:root}));
    assert.ok(events.includes('close'));
    const staging=path.join(root,'outputs','runtime','brave_plugin_capture');
    const runs=await readdir(staging);
    assert.deepEqual(await readdir(path.join(staging,runs[0])),['failure.json']);
  });
}

function virtualTab(options={}) {
  const symbols=options.symbols || ['BTCUSDT','ETHUSDT','SOLUSDT'];
  const rows=symbols.map((symbol,index)=>({symbol,
    created:`2026-09-25 09:${String(20+index).padStart(2,'0')}:39`,status:'Working',
    ...(options.investments?{investment_text:`${options.investments[index]}.00 USDT(Isolated)`}:{})}));
  const events=[];
  let scrollTop=400,opened=null,completedDrawers=0;
  let selectionActive=false,tooltipBlocked=false;
  const visible=()=>{
    const selected=scrollTop<200?rows.slice(0,2):rows.slice(1);
    return selected.map(row=>({...row,created:options.changedAfterDrawer&&completedDrawers>0
      && row.symbol==='ETHUSDT'?'2026-09-25 10:00:00':row.created}));
  };
  const scroller={
    count:async()=>1,
    evaluate:async()=>({left:0,right:500,top:150,bottom:550,
      scrollTop,clientHeight:400,scrollHeight:800}),
  };
  const details={
    count:async()=>1,
    evaluate:async()=>({x:100,y:300,hit:true,describedBy:'details-tooltip',
      rowCount:1,buttonCount:4,detailsCount:1,viewport:{width:1200,height:700},
      anchor:{x:1100,y:40,selectionCollapsed:!selectionActive},
      table:{left:0,right:500,top:150,bottom:550}}),
  };
  const drawer={
    count:async()=>opened?1:0,
    isVisible:async()=>Boolean(opened),
    locator:()=>({count:async()=>1,click:async()=>{
      events.push(`close:${opened.symbol}`);opened=null;completedDrawers++;
    }}),
    waitFor:async({state})=>{
      if ((state==='visible')!==Boolean(opened)) throw new Error('drawer wait timed out');
    },
    getByText:()=>({waitFor:async()=>{}}),
    innerText:async()=>valid.replace('BTCUSDT',opened.symbol).replace(created,opened.created)
      .replace('Invested Margin (USDT)\n',`Invested Margin (USDT)\n${options.investments?.[rows.findIndex(row=>row.symbol===opened.symbol)]||100}\n`)
      .replace('Strategy Number\n123',`Strategy Number\n${1000+rows.findIndex(row=>row.symbol===opened.symbol)}`),
  };
  const table={
    locator:()=>scroller,
    getByRole:()=>({
      evaluateAll:async()=>{
        if(completedDrawers===rows.length)options.onFinalRoster?.();
        return visible();
      },
      filter:({hasText})=>({
        count:async()=>visible().filter(row=>hasText.test(`${row.symbol} PerpNeutral 10x`)).length,
        getByRole:()=>({last:()=>({getByRole:()=>({count:async()=>4}),
          locator:()=>{details.symbol=visible().find(row=>
            hasText.test(`${row.symbol} PerpNeutral 10x`))?.symbol;return details;}})}),
      }),
    }),
  };
  const tab={
    url:async()=> 'https://www.binance.com/en/trading-bots/futures/grid/BTCUSDT',
    getScreenshot:async()=>{events.push('screenshot');},
    getAXState:async()=>{events.push('state-read');},
    click:async(point)=>{if(point[0]===1100&&point[1]===40){selectionActive=false;events.push('origin-click');return;}
        opened=rows.find(row=>row.symbol===details.symbol);
        events.push(`open:${opened.symbol}`);},
    scroll:async (_point,direction,pages)=>{
        const scrollY=direction==='down'?pages:-pages;
        events.push(`scroll:${scrollY}`);
        if (!options.stuckScroll) scrollTop=scrollY<0?0:400;
    },
    drag:async()=>{tooltipBlocked=Boolean(options.selectionBlocksTooltip&&selectionActive);
      selectionActive=true;events.push('details-hover');},
    playwright:{
      evaluate:async()=>({width:1200,height:700,x:1100,y:40,selectionCollapsed:!selectionActive}),
      getByRole:(role)=>{
        if(role==='dialog')return drawer;
        if(role==='tooltip')return {waitFor:async()=>{if(tooltipBlocked)throw new Error('selected-origin tooltip missing');events.push('tooltip');},
          count:async()=>tooltipBlocked?0:1,isVisible:async()=>!tooltipBlocked,
          evaluateAll:async()=>({count:tooltipBlocked?0:1,visible:!tooltipBlocked,id:'details-tooltip',text:'View Details'}),
          evaluate:async()=>({id:'details-tooltip',text:'View Details'})};
        if(role==='tab')return {
          allTextContents:async()=>[`UM Grid (${symbols.length+(options.incomplete?1:0)})`],
          evaluateAll:async()=>{
            if(completedDrawers===rows.length)options.onFinalRoster?.();
            return {tabs:[{text:`UM Grid (${symbols.length+(options.incomplete?1:0)})`,selected:'true'},
              {text:'Running',selected:'true'}],tableCount:1,rows:visible()};
          },
          evaluate:async()=> 'true',
          click:async()=>{events.push('focus-grid');},
        };
        if(role==='table')return table;
        throw new Error(`unexpected role ${role}`);
      },
    },
  };
  return {tab,events};
}

test('selection left by the first drawer cannot suppress the second strategy tooltip',async t=>{
  const root=await temporaryRoot(t),{tab,events}=virtualTab({symbols:['SANDUSDT','OPGUSDT'],selectionBlocksTooltip:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='origin-click').length,2);
  assert.ok(events.includes('open:OPGUSDT'));
});

test('filtered capture retains complete full-roster fences but opens only matching investments',async t=>{
  const root=await temporaryRoot(t),{tab,events}=virtualTab({investments:[1000,400,50]});
  const captured=await captureDrawers(tab,{workspaceRoot:root,investmentAmounts:[1000,400]});
  const bundle=JSON.parse(await readFile(captured.bundlePath,'utf8'));
  assert.deepEqual(captured.symbols,['BTCUSDT','ETHUSDT']);
  assert.equal(bundle.capture_scope.full_working_count,3);
  assert.deepEqual(bundle.capture_scope.full_roster_before_rows,bundle.capture_scope.full_roster_after_rows);
  assert.deepEqual(events.filter(e=>e.startsWith('open:')),['open:BTCUSDT','open:ETHUSDT']);
});

test('explicit symbol capture opens only requested drawers with full before/after fences',async t=>{
  const root=await temporaryRoot(t),{tab,events}=virtualTab();
  const captured=await captureDrawers(tab,{workspaceRoot:root,symbols:['SOLUSDT','BTCUSDT']});
  const bundle=JSON.parse(await readFile(captured.bundlePath,'utf8'));
  assert.deepEqual(captured.symbols,['BTCUSDT','SOLUSDT']);
  assert.equal(bundle.capture_scope.type,'symbols');
  assert.equal(bundle.capture_scope.full_working_count,3);
  assert.deepEqual(bundle.capture_scope.full_roster_before_rows,bundle.capture_scope.full_roster_after_rows);
  assert.deepEqual(events.filter(e=>e.startsWith('open:')),['open:BTCUSDT','open:SOLUSDT']);
});

test('virtual table captures every Working bot from a complete overlapping sweep',async t=>{
  const root=await temporaryRoot(t),{tab,events}=virtualTab();
  const result=await captureDrawers(tab,{workspaceRoot:root});
  const bundle=JSON.parse(await readFile(result.bundlePath,'utf8'));
  assert.deepEqual(result.symbols,['BTCUSDT','ETHUSDT','SOLUSDT']);
  assert.deepEqual(bundle.roster_before_rows,bundle.roster_after_rows);
  assert.deepEqual(bundle.captures.map(c=>c.symbol),result.symbols);
  assert.deepEqual(events.filter(e=>e.startsWith('open:')),
    ['open:BTCUSDT','open:ETHUSDT','open:SOLUSDT']);
  assert.ok(bundle.captures.every(c=>c.raw_text_path.startsWith(path.join(root,'Live')+path.sep)));
});

test('SUSDT drawer row does not collide with KASUSDT',async t=>{
  const root=await temporaryRoot(t);
  const {tab,events}=virtualTab({symbols:['KASUSDT','SUSDT','SOLUSDT']});
  const result=await captureDrawers(tab,{workspaceRoot:root});
  assert.deepEqual(result.symbols,['KASUSDT','SUSDT','SOLUSDT']);
  assert.deepEqual(events.filter(e=>e.startsWith('open:')),
    ['open:KASUSDT','open:SUSDT','open:SOLUSDT']);
});

test('27-strategy fresh DOM capture retains every native ID, creation time and roster fence',async t=>{
  const root=await temporaryRoot(t);
  const symbols=Array.from({length:27},(_,i)=>`T${i}USDT`);
  const {tab,events}=virtualTab({symbols});
  tab.playwright.domSnapshot=async()=> 'fresh rendered DOM';
  const result=await captureDrawers(tab,{workspaceRoot:root});
  const bundle=JSON.parse(await readFile(result.bundlePath,'utf8'));
  assert.equal(bundle.captures.length,27);
  assert.equal(new Set(bundle.captures.map(row=>row.strategy_id)).size,27);
  assert.deepEqual(bundle.roster_before_rows,bundle.roster_after_rows);
  assert.deepEqual(bundle.captures.map(row=>row.symbol),symbols);
  assert.equal(events.filter(e=>e.startsWith('open:')).length,27);
  for(const [i,cap] of bundle.captures.entries()){
    assert.equal(cap.strategy_id,String(1000+i));
    assert.equal(cap.deployment_time_lima,bundle.roster_before_rows[i].created);
    assert.equal(parseDrawer(await readFile(cap.raw_text_path,'utf8'),cap.symbol,cap.deployment_time_lima).strategyId,cap.strategy_id);
  }
});

test('final roster sweep cannot publish a bundle after the bounded acquisition deadline',async t=>{
  const root=await temporaryRoot(t);
  const originalNow=Date.now,baseline=originalNow();let extra=0;
  const {tab}=virtualTab({onFinalRoster:()=>{extra=601000;}});
  Date.now=()=>baseline+extra;
  try{await assert.rejects(captureDrawers(tab,{workspaceRoot:root}),/capture deadline exceeded/);}
  finally{Date.now=originalNow;}
  const staging=path.join(root,'outputs','runtime','brave_plugin_capture');
  const runs=await readdir(staging);
  assert.deepEqual(await readdir(path.join(staging,runs[0])),['failure.json']);
});

for (const options of [{incomplete:true},{stuckScroll:true},{changedAfterDrawer:true}]) {
  test(`virtual table fails closed on ${JSON.stringify(options)}`,async t=>{
    const root=await temporaryRoot(t),{tab,events}=virtualTab(options);
    await assert.rejects(captureDrawers(tab,{workspaceRoot:root}));
    if (options.incomplete || options.stuckScroll) {
      assert.ok(!events.some(e=>e.startsWith('open:')));
    }
    const staging=path.join(root,'outputs','runtime','brave_plugin_capture');
    const runs=await readdir(staging);
    assert.deepEqual(await readdir(path.join(staging,runs[0])),['failure.json']);
  });
}
