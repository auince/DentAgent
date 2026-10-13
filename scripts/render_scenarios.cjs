// Render code-native chart layouts and CSS camera effects with a headless browser.
const fs=require('fs'),path=require('path');
const {chromium}=require(process.env.MEDAGENT_PLAYWRIGHT || 'playwright');
(async()=>{
 const root=path.resolve(__dirname,'..');const cases=JSON.parse(fs.readFileSync(path.join(root,'scenarios/implicit_v1/cases.json')));
 const browser=await chromium.launch({headless:true,executablePath:process.env.MEDAGENT_BROWSER});
 const page=await browser.newPage({viewport:{width:1200,height:1520},deviceScaleFactor:1});
 for(const c of cases.filter(c=>c.modality==='image_only')){
  await page.goto('file://'+path.join(root,c.html));await page.evaluate(()=>document.fonts.ready);
  fs.writeFileSync(path.join(root,c.image).replace(/\.png$/,'.ocr.txt'),await page.locator('article').innerText());
  await page.screenshot({path:path.join(root,c.image),fullPage:true});
 }
 await browser.close();console.log('Rendered 30 image-only chart inputs');
})().catch(e=>{console.error(e);process.exit(1)});
