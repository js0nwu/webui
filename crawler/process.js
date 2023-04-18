import axios from 'axios';
import puppeteer from 'puppeteer';
import fs from 'fs';
import stringSimilarity from "string-similarity";
import zlib from 'zlib';

import AWS from 'aws-sdk';

//configuring the AWS environment
AWS.config.update({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
});

//upload files to AWS s3
async function uploadFile(file, enable_uploads = true) {
  if (!enable_uploads) {
    return;
  }
  if (process.env.S3_ENDPOINT) {
    var s3 = new AWS.S3({
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
      endpoint: process.env.S3_ENDPOINT,
      s3ForcePathStyle: true,
      signatureVersion: 'v4'
    });
  } else {
    var s3 = new AWS.S3()
  }
  var filePath = file;

  //configuring parameters
  var params = {
    Bucket: 'biglab-ui-bucket',
    Body: fs.createReadStream(filePath),
    Key: filePath
  };

  s3.upload(params, function (err, data) {
    //handle error
    if (err) {
      console.log("Error", err);
    }

    //success
    if (data) {
      // console.log("Uploaded in:", data.Location);
      fs.unlink(file, (err) => {
        if (err) throw err;
        // console.log(file + 'was deleted');
      });
    }
  });
}


const RETRY_TIMEOUT = 10000
const PAGE_TIMEOUT = 360000
const LOAD_TIME = 1000

const OUTPUT_DIR = "crawls"

if (!fs.existsSync(OUTPUT_DIR)) {
  fs.mkdirSync(OUTPUT_DIR);
}

async function processPage(browser, url) {
  /**
   * @param {import("puppeteer").Page} page
   */
  function getClient(page) {
    return /** @type {import('puppeteer').CDPSession} */ (
      /** @type {any} */ (page)._client
    )
  }

  /**
   * @param {import("puppeteer").CDPSession} client
   */
  async function getAccessibilityTree(client) {
    return /** @type {import("puppeteer/lib/esm/protocol").default.Accessibility.getFullAXTreeReturnValue} */ (await client.send(
      'Accessibility.getFullAXTree',
    ))
  }

  /**
   * @param {import("puppeteer").Frame} frame
   * @param {number} backendNodeId
   */
  async function resolveNodeFromBackendNodeId(frame, backendNodeId) {
    const ctx = await Promise.resolve(frame.executionContext())
    return /** @type {import('puppeteer').ElementHandle} */ (
      /** @type {any} */ (ctx)._adoptBackendNodeId(backendNodeId)
    )
  }

  const key = OUTPUT_DIR + "/" + Math.floor(Date.now());
  if (!fs.existsSync(key)) {
    fs.mkdirSync(key);
  }

  async function loadAndDismiss(page) {
    // load and dismiss
    await new Promise(resolve => setTimeout(resolve, LOAD_TIME));
    // dismiss cookie popups https://stackoverflow.com/a/65008319
    await page.evaluate(_ => {
      function xcc_contains(selector, text) {
        var elements = document.querySelectorAll(selector);
        return Array.prototype.filter.call(elements, function (element) {
          return RegExp(text, "i").test(element.textContent.trim());
        });
      }
      var _xcc;
      _xcc = xcc_contains('[id*=cookie] a, [class*=cookie] a, [id*=cookie] button, [class*=cookie] button', '^(Accept|Accept all|I understand|Agree|Okay|OK)$');
      if (_xcc != null && _xcc.length != 0) { _xcc[0].click(); }
    });
  }

  async function captureDataAndGetLinks(page, devString) {
    // save screenshot
    var deviceScreenshotPath = key + "/" + devString + "-screenshot.webp"
    await page.screenshot({ path: deviceScreenshotPath, quality: 50 });
    uploadFile(deviceScreenshotPath);

    // save screenshot (full) // takes too much storage
    var deviceFullScreenshotPath = key + "/" + devString + "-screenshot-full.webp"
    await page.screenshot({ path: deviceFullScreenshotPath, fullPage: true, quality: 50 });
    uploadFile(deviceFullScreenshotPath);

    // accessibility tree
    var deviceAxTreePath = key + "/" + devString + "-axtree.json"
    const client = getClient(page)
    const snapshot = await getAccessibilityTree(client)
    // fs.writeFileSync(deviceAxTreePath, JSON.stringify(snapshot))
    zlib.gzip(JSON.stringify(snapshot), function (err, binary) {
      fs.writeFileSync(deviceAxTreePath + ".gz", binary)
      uploadFile(deviceAxTreePath + ".gz")
    })

    // bounding boxes
    var deviceBBPath = key + "/" + devString + "-bb.json"
    var bbNodes = {}
    // boxes
    var deviceBoxPath = key + "/" + devString + "-box.json"
    var boxNodes = {}
    // classes
    var deviceClassPath = key + "/" + devString + "-class.json"
    var classNodes = {}
    // styles
    var deviceStylePath = key + "/" + devString + "-style.json"
    var styleNodes = {}
    // viewport
    var deviceViewportPath = key + "/" + devString + "-viewport.json"
    var viewportNodes = {}
    for (var j = 0; j < snapshot.nodes.length; j++) {
      var axNode = snapshot.nodes[j]
      if (axNode.backendDOMNodeId === undefined) {
        continue
      }
      const domHandle = await resolveNodeFromBackendNodeId(page.mainFrame(), axNode.backendDOMNodeId)
      const bbNode = await domHandle.boundingBox()
      bbNodes[axNode.backendDOMNodeId] = bbNode
      const boxNode = await domHandle.boxModel()
      boxNodes[axNode.backendDOMNodeId] = boxNode
      const classNode = await domHandle.getProperty("classList")
      const classNodeJSON = await classNode.jsonValue()
      classNodes[axNode.backendDOMNodeId] = classNodeJSON

      try {
        const styleNode = await domHandle.evaluate((el) => {
          const computedStyle = getComputedStyle(el);
          return [...computedStyle].reduce((elementStyles, property) => ({ ...elementStyles, [property]: computedStyle.getPropertyValue(property) }), {})
        });
        styleNodes[axNode.backendDOMNodeId] = styleNode

        const viewportIntersect = await domHandle.isIntersectingViewport()
        viewportNodes[axNode.backendDOMNodeId] = viewportIntersect
      } catch (error) {

      }

    }
    // fs.writeFileSync(deviceBBPath, JSON.stringify(bbNodes))
    zlib.gzip(JSON.stringify(bbNodes), function (err, binary) {
      fs.writeFileSync(deviceBBPath + ".gz", binary)
      uploadFile(deviceBBPath + ".gz")
    })
    // fs.writeFileSync(deviceBoxPath, JSON.stringify(boxNodes))
    zlib.gzip(JSON.stringify(boxNodes), function (err, binary) {
      fs.writeFileSync(deviceBoxPath + ".gz", binary)
      uploadFile(deviceBoxPath + ".gz")
    })
    // fs.writeFileSync(deviceClassPath, JSON.stringify(classNodes))
    zlib.gzip(JSON.stringify(classNodes), function (err, binary) {
      fs.writeFileSync(deviceClassPath + ".gz", binary)
      uploadFile(deviceClassPath + ".gz")
    })

    // fs.writeFileSync(deviceStylePath, JSON.stringify(styleNodes))
    zlib.gzip(JSON.stringify(styleNodes), function (err, binary) {
      fs.writeFileSync(deviceStylePath + ".gz", binary)
      uploadFile(deviceStylePath + ".gz")
    })
    // fs.writeFileSync(deviceViewportPath, JSON.stringify(viewportNodes))
    zlib.gzip(JSON.stringify(viewportNodes), function (err, binary) {
      fs.writeFileSync(deviceViewportPath + ".gz", binary)
      uploadFile(deviceViewportPath + ".gz")
    })

    // lighthouse results (nvm, too slow)
    // html
    const deviceHTMLPath = key + "/" + devString + "-html.html"
    const pageHTML = await page.content()
    fs.writeFileSync(deviceHTMLPath, pageHTML)
    uploadFile(deviceHTMLPath)
    // file with full URL
    const deviceURLPath = key + "/" + devString + "-url.txt"
    fs.writeFileSync(deviceURLPath, url)
    uploadFile(deviceURLPath)

    // links
    const deviceHrefsPath = key + "/" + devString + "-links.json"
    const hrefs = await page.$$eval('a', as => as.map(a => a.href));
    fs.writeFileSync(deviceHrefsPath, JSON.stringify(hrefs))
    uploadFile(deviceHrefsPath)

    const pageLinks = new Set()
    for (var h = 0; h < hrefs.length; h++) {
      const href = hrefs[h]
      pageLinks.add(href)
    }
    return pageLinks
  }

  var devices = ["iPhone 13 Pro", "iPad Pro"]
  // https://gs.statcounter.com/screen-resolution-stats/desktop/worldwide
  var resolutions = [[1920, 1080], [1366, 768], [1536, 864], [1280, 720]]

  var links = new Set()

  var page = await browser.newPage();
  await page.setUserAgent(
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4182.0 Safari/537.36"
  );

  await page.goto(url)
  await loadAndDismiss(page);

  var devString = "default";

  for (var i = 0; i < resolutions.length; i++) {
    const res = resolutions[i]
    // https://stackoverflow.com/a/52561280
    await page.setViewport({ width: res[0], height: res[1] })
    const plinks = await captureDataAndGetLinks(page, devString + "_" + res[0] + "-" + res[1]);
    links = new Set([...links, ...plinks])
  }

  for (var i = 0; i < devices.length; i++) {
    var dev = devices[i]
    devString = dev.replace(" ", "-")

    page = await browser.newPage();
    const m = puppeteer.devices[dev]
    await page.emulate(m);
    await page.goto(url);
    await loadAndDismiss(page);
    const plinks = await captureDataAndGetLinks(page, devString);
    links = new Set([...links, ...plinks]);
  }

  return links;
}

const serverAddURL = process.env.SERVER_URL + "/add_url";
const serverRequestURL = process.env.SERVER_URL + "/request_url";

(async () => {


  while (true) {
    // console.log("start")
    var browser = await puppeteer.launch({ args: ['--single-process', '--no-zygote', '--no-sandbox'] });
    try {
      const res = await axios.get(serverRequestURL).catch((error) => {
        console.log(error);
      });
      if (res.data.url === null) {
        await new Promise(resolve => setTimeout(resolve, RETRY_TIMEOUT));
        console.log("no url")
        continue;
      }
      const url = res.data.url
      // console.log("URL")
      console.log(url)
      async function visitURL() {
        // add a time out, add visited set, block list, robots.txt
        const links = await processPage(browser, url);
        // console.log(links)
        var linksList = Array.from(links);

        const stringFilter = (source, rate = 0.85) => {
          let _source, matches, x, y;
          _source = source.slice();
          matches = [];
          for (x = _source.length - 1; x >= 0; x--) {
            let output = _source.splice(x, 1);

            for (y = _source.length - 1; y >= 0; y--) {
              var match = stringSimilarity.compareTwoStrings(output[0], _source[y]);
              // console.log(output[0], _source[y], match);
              if (match > rate) {
                output.push(_source[y]);
                _source.splice(y, 1);
                x--;
              }
            }
            matches.push(output);
          }
          return matches;
        };

        let output = stringFilter(linksList);

        const linksListPruned = []
        for (var s = 0; s < output.length; s++) {
          const group = output[s]
          if (group.length > 0) {
            const randomElement = group[Math.floor(Math.random() * group.length)];
            linksListPruned.push(randomElement)
          }
        }
        linksList = linksListPruned

        for (var l = 0; l < linksList.length; l++) {
          const link = linksList[l]
          axios.post(serverAddURL, {
            url: link
          }, {
            headers: {
              "Content-Type": "application/json"
            }
          })
            .then(response => {
              // console.log(response.data);
            }).catch((error) => {
              console.log(error);
            });
        }
      }

      const timeout = (cb, interval) => () =>
        new Promise(resolve => setTimeout(() => cb(resolve), interval))

      const onTimeout = timeout(resolve =>
        resolve("Timeout exceeded"), PAGE_TIMEOUT)

      await Promise.race([visitURL, onTimeout].map(f => f()))
      // console.log("done")

    } catch (error) {
      console.log(error)

      await new Promise(resolve => setTimeout(resolve, RETRY_TIMEOUT));
      // continue
    } finally {
      await browser.close();
    }
  }
})();
