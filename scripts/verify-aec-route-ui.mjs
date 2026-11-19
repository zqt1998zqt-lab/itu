import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';
const require=createRequire(import.meta.url);
const {chromium}=require('/home/zhangqingtian/.npm/_npx/9833c18b2d85bc59/node_modules/playwright');
const root='/home/zhangqingtian/project/itu';
const artifacts=fs.mkdtempSync(path.join(root,'build/aec-route-ui-'));
const chrome='/home/zhangqingtian/.cache/ms-playwright/chromium_headless_shell-1234/chrome-linux/headless_shell';
const stderr=fs.openSync(path.join(artifacts,'chromium.log'),'ax');
// Start Chromium ourselves with a persistent profile; no cleanup wrapper.
const child=spawn(chrome,['--no-sandbox','--disable-dev-shm-usage','--headless',
 '--remote-debugging-port=0','--user-data-dir='+path.join(artifacts,'profile'),
 '--window-size=1440,1050','about:blank'],{stdio:['ignore','ignore',stderr]});
const portFile=path.join(artifacts,'profile','DevToolsActivePort');
for(let i=0;i<100&&!fs.existsSync(portFile);i++)await new Promise(r=>setTimeout(r,100));
assert(fs.existsSync(portFile),'Chrome did not expose CDP');
const port=fs.readFileSync(portFile,'utf8').split('\n')[0];
const browser=await chromium.connectOverCDP('http://127.0.0.1:'+port);
const context=browser.contexts()[0];
const page=await context.newPage();
const errors=[];
page.on('pageerror',e=>errors.push(e.message));
const result={artifacts,checks:[]};
try{
 await page.setViewportSize({width:1440,height:1050});
 await page.goto('http://127.0.0.1:5173/#alignment',{waitUntil:'domcontentloaded'});
 const selection=page.locator('select').filter({has:page.locator('option[value="speex_fused_dsp"]')}).first();
 await selection.waitFor({timeout:30000});
 await selection.selectOption('speex_fused_dsp');
 await page.getByText(/DSP 路由不可用：.*HiFi SDK\/provider/).first().waitFor({timeout:20000});
 await page.screenshot({path:path.join(artifacts,'alignment-dsp-unavailable.png'),fullPage:false});
 result.checks.push('alignment DSP selection displays actual unavailable reason');
 await selection.selectOption('speex_fused_cpu');
 await page.waitForTimeout(150);
 assert.equal(await selection.inputValue(),'speex_fused_cpu');
 result.checks.push('alignment CPU selection remains distinct');
 await page.goto('http://127.0.0.1:5173/#enhancement',{waitUntil:'domcontentloaded'});
 const aecSelect=page.locator('select[aria-label="选择AEC 回声消除配置"]');
 await aecSelect.waitFor({timeout:20000});
 await aecSelect.selectOption('speex_fused_dsp');
 await page.getByText(/DSP 路由不可用：.*HiFi SDK\/provider/).first().waitFor({timeout:20000});
 const submit=page.getByRole('button',{name:'生成发送处理输出（TX）',exact:true});
 if(await submit.count()) {
   const pcm=Buffer.alloc(44+16000*2);
   pcm.write('RIFF',0);pcm.writeUInt32LE(pcm.length-8,4);pcm.write('WAVEfmt ',8);
   pcm.writeUInt32LE(16,16);pcm.writeUInt16LE(1,20);pcm.writeUInt16LE(1,22);
   pcm.writeUInt32LE(16000,24);pcm.writeUInt32LE(32000,28);
   pcm.writeUInt16LE(2,32);pcm.writeUInt16LE(16,34);pcm.write('data',36);
   pcm.writeUInt32LE(32000,40);
   for(let i=0;i<16000;i++)pcm.writeInt16LE(Math.round(2000*Math.sin(2*Math.PI*440*i/16000)),44+i*2);
   const inputs=page.locator('.audio-upload-grid input[type="file"]');
   await inputs.nth(0).setInputFiles({name:'route-mic.wav',mimeType:'audio/wav',buffer:pcm});
   await inputs.nth(1).setInputFiles({name:'route-far.wav',mimeType:'audio/wav',buffer:pcm});
   assert(await submit.isDisabled(),'DSP unavailable must block a populated form');
   await aecSelect.selectOption('speex_fused_cpu');
   await page.waitForTimeout(150);
   assert(await submit.isEnabled(),'CPU selection must restore submit for populated form');
   result.checks.push('populated form blocks unavailable DSP and enables CPU');
   await aecSelect.selectOption('speex_fused_dsp');
 }
 await page.screenshot({path:path.join(artifacts,'chain-dsp-unavailable.png'),fullPage:false});
 result.checks.push('chain AEC DSP option displays actual unavailable reason');
 result.pageErrors=errors;
 assert.equal(errors.length,0,errors.join('\n'));
 fs.writeFileSync(path.join(artifacts,'result.json'),JSON.stringify(result,null,2),{flag:'wx'});
 console.log(JSON.stringify(result));
}catch(error){
 result.error=String(error);result.pageErrors=errors;
 await page.screenshot({path:path.join(artifacts,'failure.png'),fullPage:false}).catch(()=>{});
 fs.writeFileSync(path.join(artifacts,'failure.json'),JSON.stringify(result,null,2),{flag:'wx'});
 throw error;
}finally{
 await browser.close();
 child.kill('SIGTERM');
}
