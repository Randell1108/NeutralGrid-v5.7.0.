import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, readdir } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { captureDrawers, parseDrawer, validateRoster } from '../scripts/capture_recurring_drawers.mjs';

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
  let opened = false, y = options.tableInitiallyOffscreen ? 1000 : options.nearBottom ? -19.6 : options.occluded ? 329 : 800;
  let tableTop = options.tableInitiallyOffscreen ? 950 : options.nearBottom ? 203.2 : options.occluded ? 616 : 100;
  let rosterReads = 0, discoveryReads = 0, closeClicks = 0;
  const details = {
    count:async()=>1,
    evaluate:async()=>({x:100,y,hit:y>=tableTop&&y<=(options.nearBottom?634.2:tableTop+425)&&y<=660,
      table:{left:0,right:500,top:tableTop,bottom:options.nearBottom?634.2:tableTop+425}}),
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
    locator:()=>({count:async()=>1,click:async()=>{
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
    cua:{
      click:async()=>{
        events.push('details-click');opened=true;
        if(options.clickResponseLost)throw new Error('click response lost');
      },
      scroll:async p=>{events.push(p);if(!options.stuckScroll && !options.tableInitiallyOffscreen){
        if(tableTop>560){tableTop=150;y=-137;}else y=300;
      } if(options.scrollResponseLost) throw new Error('scroll response lost');},
      move:async()=>{},
    },
    playwright:{
      evaluate:async()=>({width:1200,height:700}),
      getByRole:(role)=>{
        if(role==='dialog')return drawer;
        if(role==='tooltip')return {
          count:async()=>options.noTooltip?0:1,
          isVisible:async()=>!options.noTooltip,
          waitFor:async()=>{if(options.noTooltip)throw new Error('no tooltip');events.push('tooltip');},
        };
        if(role==='tab')return {
          allTextContents:async()=>[`UM Grid (${options.partial?2:1})`],
          evaluate:async()=>options.wrongTab?'false':'true',
          click:async()=>{events.push('focus-grid'); if(options.tableInitiallyOffscreen){tableTop=150;y=300;}},
        };
        if(role==='table')return {getByRole:()=>({
          filter:()=>rowLocator,
          evaluateAll:async()=>{
            rosterReads++;
            return [{...row,created:options.changedRoster&&rosterReads>3?'2026-09-25 10:00:00':created}];
          },
        })};
        throw new Error(`unexpected role ${role}`);
      },
    },
  };
  return {tab,events};
}

async function temporaryRoot(t) {
  const root = await mkdtemp(path.join(os.tmpdir(),'neutralgrid-drawer-test-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  return root;
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
test('transient read-only drawer check is retried before capture',async t=>{
  const root=await temporaryRoot(t),{tab,events}=fakeTab({transientDrawerCount:true});
  assert.equal((await captureDrawers(tab,{workspaceRoot:root})).status,'complete');
  assert.equal(events.filter(e=>e==='details-click').length,1);
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

for (const options of [{noTooltip:true},{stuckScroll:true},{partial:true},{wrongTab:true}]) {
  test(`unsafe acquisition stops before any drawer click: ${JSON.stringify(options)}`,async t=>{
    const root=await temporaryRoot(t),{tab,events}=fakeTab(options);
    await assert.rejects(captureDrawers(tab,{workspaceRoot:root}));
    assert.ok(!events.includes('details-click'));
    const staging=path.join(root,'outputs','runtime','chrome_plugin_capture');
    const runs=await readdir(staging);
    assert.deepEqual(await readdir(path.join(staging,runs[0])),['failure.json']);
  });
}

for (const options of [{changedRoster:true},{invalidDrawer:true}]) {
  test(`incomplete capture closes drawer and publishes failure only: ${JSON.stringify(options)}`,async t=>{
    const root=await temporaryRoot(t),{tab,events}=fakeTab(options);
    await assert.rejects(captureDrawers(tab,{workspaceRoot:root}));
    assert.ok(events.includes('close'));
    const staging=path.join(root,'outputs','runtime','chrome_plugin_capture');
    const runs=await readdir(staging);
    assert.deepEqual(await readdir(path.join(staging,runs[0])),['failure.json']);
  });
}

function virtualTab(options={}) {
  const symbols=['BTCUSDT','ETHUSDT','SOLUSDT'];
  const rows=symbols.map((symbol,index)=>({symbol,
    created:`2026-09-25 09:2${index}:39`,status:'Working'}));
  const events=[];
  let scrollTop=400,opened=null,completedDrawers=0;
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
    evaluate:async()=>({x:100,y:300,hit:true,
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
    innerText:async()=>valid.replace('BTCUSDT',opened.symbol).replace(created,opened.created),
  };
  const table={
    locator:()=>scroller,
    getByRole:()=>({
      evaluateAll:async()=>visible(),
      filter:({hasText})=>({
        count:async()=>visible().filter(row=>hasText.includes(`${row.symbol} Perp`)).length,
        getByRole:()=>({last:()=>({getByRole:()=>({count:async()=>4}),
          locator:()=>{details.symbol=hasText.split(' ')[0];return details;}})}),
      }),
    }),
  };
  const tab={
    url:async()=> 'https://www.binance.com/en/trading-bots/futures/grid/BTCUSDT',
    cua:{
      click:async()=>{opened=rows.find(row=>row.symbol===details.symbol);
        events.push(`open:${opened.symbol}`);},
      scroll:async ({scrollY})=>{
        events.push(`scroll:${scrollY}`);
        if (!options.stuckScroll) scrollTop=scrollY<0?0:400;
      },
      move:async()=>{},
    },
    playwright:{
      evaluate:async()=>({width:1200,height:700}),
      getByRole:(role)=>{
        if(role==='dialog')return drawer;
        if(role==='tooltip')return {waitFor:async()=>{events.push('tooltip');}};
        if(role==='tab')return {
          allTextContents:async()=>[`UM Grid (${options.incomplete?4:3})`],
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

for (const options of [{incomplete:true},{stuckScroll:true},{changedAfterDrawer:true}]) {
  test(`virtual table fails closed on ${JSON.stringify(options)}`,async t=>{
    const root=await temporaryRoot(t),{tab,events}=virtualTab(options);
    await assert.rejects(captureDrawers(tab,{workspaceRoot:root}));
    if (options.incomplete || options.stuckScroll) {
      assert.ok(!events.some(e=>e.startsWith('open:')));
    }
    const staging=path.join(root,'outputs','runtime','chrome_plugin_capture');
    const runs=await readdir(staging);
    assert.deepEqual(await readdir(path.join(staging,runs[0])),['failure.json']);
  });
}
