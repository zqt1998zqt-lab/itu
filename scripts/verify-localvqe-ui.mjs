import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';
const require=createRequire(import.meta.url);
const {chromium}=require('/home/zhangqingtian/.npm/_npx/9833c18b2d85bc59/node_modules/playwright');
const root='/home/zhangqingtian/project/itu';
const artifacts=fs.mkdtempSync(path.join(root,'build/localvqe-ui-'));
const chrome='/home/zhangqingtian/.cache/ms-playwright/chromium_headless_shell-1234/chrome-linux/headless_shell';
const stderr=fs.openSync(path.join(artifacts,'chromium.log'),'ax');
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
const errors=[];page.on('pageerror',e=>errors.push(e.message));
const consoleErrors=[];page.on('console',e=>{if(e.type()==='error')consoleErrors.push(e.text());});
const result={artifacts,checks:[]};
try{
 await page.setViewportSize({width:1440,height:1050});
 await page.goto('http://127.0.0.1:5173/#localvqe',{waitUntil:'domcontentloaded'});
 await page.locator('#localvqe-title').waitFor({timeout:30000});
 await page.getByText('当前入口使用 LocalVQE 自己的 DAF 前端').first().waitFor({timeout:20000});
 const options=await page.locator('select[aria-label="LocalVQE 权重"] option').allTextContents();
 assert.equal(options.length,4,JSON.stringify(options));
 assert(options.some(o=>o.includes('203K'))&&options.some(o=>o.includes('49K')));
 result.checks.push('page renders four available pinned weights and explicit scope boundary');
 const micPath=path.join(root,'datasets/itu-vehicle-fields-float-20261010/cases/field-p1-m1-diffuse-snr30/before.wav');
 const refPath=path.join(root,'datasets/itu-vehicle-fields-float-20261010/cases/field-p1-m1-diffuse-snr30/ref.wav');
 function headWav(source,seconds=1){
   const rate=16000,samples=seconds*rate,bytes=44+samples*4;const out=Buffer.alloc(bytes);
   const sourceBytes=fs.readFileSync(source);
   assert.equal(sourceBytes.toString('ascii',0,4),'RIFF');
   assert.equal(sourceBytes.toString('ascii',8,12),'WAVE');
   // WAV metadata chunks vary: locate fmt/data by their declared sizes.
   let offset=12,dataStart=-1,format;
   while(offset+8<=sourceBytes.length){
     const chunk=sourceBytes.toString('ascii',offset,offset+4);
     const size=sourceBytes.readUInt32LE(offset+4);
     if(chunk==='fmt ')format={encoding:sourceBytes.readUInt16LE(offset+8),channels:sourceBytes.readUInt16LE(offset+10),rate:sourceBytes.readUInt32LE(offset+12),bits:sourceBytes.readUInt16LE(offset+22)};
     if(chunk==='data'){dataStart=offset+8;break;}
     offset+=8+size+(size%2);
   }
   assert.deepEqual(format,{encoding:3,channels:1,rate:16000,bits:32});
   assert(dataStart>0&&sourceBytes.length>=dataStart+samples*4,'complete source PCM is required');
   sourceBytes.copy(out,44,dataStart,dataStart+samples*4);
   out.write('RIFF',0);out.writeUInt32LE(bytes-8,4);out.write('WAVEfmt ',8);
   out.writeUInt32LE(16,16);out.writeUInt16LE(3,20);out.writeUInt16LE(1,22);
   out.writeUInt32LE(rate,24);out.writeUInt32LE(rate*4,28);out.writeUInt16LE(4,32);
   out.writeUInt16LE(32,34);out.write('data',36);out.writeUInt32LE(samples*4,40);
   return out;
 }
 const micInput=page.getByLabel('麦克风混合输入',{exact:true});const refInput=page.getByLabel('远端播放信号',{exact:true});
 await micInput.setInputFiles({name:'mic.wav',mimeType:'audio/wav',buffer:headWav(micPath)});
 await refInput.setInputFiles({name:'ref.wav',mimeType:'audio/wav',buffer:headWav(refPath)});
 await page.getByRole('button',{name:'执行 LocalVQE CPU 测试'}).click();
 await page.getByRole('heading',{name:'实际执行结果'}).waitFor({timeout:30000});
 await page.locator('audio[aria-label="LocalVQE 处理后输出"]').waitFor({timeout:10000});
 const text=await page.locator('.localvqe-result').innerText();
 assert(/process_ms/.test(text)&&/output_sha256/.test(text),text);
 result.checks.push('real browser upload executes CPU inference and exposes metrics plus audio output');
 await page.screenshot({path:path.join(artifacts,'localvqe-result.png'),fullPage:false});
 await page.locator('#localvqe-ab-title').waitFor({timeout:20000});
 const abResponse=await page.request.get('http://127.0.0.1:5173/api/localvqe/mdf-mask-ab');
 assert.equal(abResponse.status(),200);
 const ab=await abResponse.json();
 assert.equal(ab.case_count,20);assert.equal(ab.duration_s,12);
 assert.deepEqual(Object.keys(ab.categories),['near_single','far_single','double_talk','path_change']);
 assert.equal(ab.categories.double_talk.measured_case_count,4);
 assert.equal(ab.categories.double_talk.invalid_cases.length,1);
 assert.equal(await page.locator('.localvqe-ab-group').count(),4);
 const abText=await page.locator('.localvqe-ab').innerText();
 assert(abText.includes('有效 4/5 组')&&abText.includes('近端单讲')&&abText.includes('实际同帧'),abText);
 assert(!abText.includes('回声残差 RMS'),'double-talk error must not be labelled pure echo');
 for(const group of Object.values(ab.categories)){
   const url=group.cases[0].audio.mdf_residual_mask;
   const response=await page.request.get('http://127.0.0.1:5173'+url);
   assert.equal(response.status(),200);assert((await response.body()).length>192000*4);
 }
 await page.getByText('按条件组试听同一时间轴的 A/B 输出',{exact:true}).first().click();
 const nearAudio=page.locator('.localvqe-ab-group').first().locator('audio').nth(3);
 await nearAudio.evaluate(async element=>{await element.play();});
 await page.waitForTimeout(180);
 const playback=await nearAudio.evaluate(element=>({currentTime:element.currentTime,error:element.error?.message??null,paused:element.paused}));
 assert.equal(playback.error,null);assert(playback.currentTime>0,'comparison WAV must actually play');
 await nearAudio.evaluate(element=>element.pause());
 result.checks.push('20 full-duration cases, four groups, invalid double-talk and before/after path windows render');
 result.checks.push('hashed comparison audio downloads in all groups; near A/B WAV actually plays');
 result.playback=playback;
 await page.locator('#localvqe-ab-title').scrollIntoViewIfNeeded();
 await page.screenshot({path:path.join(artifacts,'localvqe-ab-summary.png'),fullPage:false});
 result.reportSha256=ab.result_sha256;
 result.pageErrors=errors;assert.equal(errors.length,0,errors.join('\n'));
 result.consoleErrors=consoleErrors;assert.equal(consoleErrors.length,0,consoleErrors.join('\n'));
 fs.writeFileSync(path.join(artifacts,'result.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify(result));
}catch(error){
 result.error=String(error);result.pageErrors=errors;
 await page.screenshot({path:path.join(artifacts,'failure.png'),fullPage:false}).catch(()=>{});
 fs.writeFileSync(path.join(artifacts,'failure.json'),JSON.stringify(result,null,2)+'\n',{flag:'wx'});
 throw error;
}finally{await browser.close();child.kill('SIGTERM');}
