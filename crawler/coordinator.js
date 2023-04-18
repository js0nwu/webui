import express from 'express';
import normalizeUrl from 'normalize-url';
import FuzzySet from 'fuzzyset'
import fs from 'fs';

var seedList = [
    normalizeUrl("https://news.ycombinator.com"),
    normalizeUrl("https://old.reddit.com"),
    normalizeUrl("https://lobste.rs"),
    normalizeUrl("https://design.google"),
    normalizeUrl("https://www.yahoo.com"),
    normalizeUrl("https://getbootstrap.com"),
    normalizeUrl("https://material.io/design"),
    normalizeUrl("https://www.nih.gov/"),
    normalizeUrl("https://www.cdc.gov/"),
    normalizeUrl("https://semantic-ui.com/"),
    normalizeUrl("https://builtwithbootstrap.com/"),
    normalizeUrl("https://github.com/hemanth/awesome-pwa/"),
    normalizeUrl("https://dribbble.com/"),
    normalizeUrl("https://news.google.com/")
]

if (fs.existsSync("urls.json")) {
    let rawdata = fs.readFileSync("urls.json");
    urlList = JSON.parse(rawdata);
    console.log("loaded from urls.json");
}

const visited = new FuzzySet()

if (fs.existsSync("visited.json")) {
    let rawdata = fs.readFileSync("visited.json");
    let visitedlist = JSON.parse(rawdata);
    for (var i = 0; i < visitedlist.length; i++) {
        visited.add(visitedlist[i]);
    }
    console.log("loaded from visited.json");
    console.log(visited.length());
}

var urlList = {};

const app = express();
app.use(express.json());

function addNewURL(addUrl) {
    console.log(addUrl);
    const newUrl = new URL(addUrl);
    visited.add(addUrl);
    var hostName = newUrl.hostname;
    if (hostName.includes(".")) {
        const hostNameSplit = hostName.split(".");
        if (hostNameSplit.length >= 2) {
            hostName = hostNameSplit[hostNameSplit.length - 2] + hostNameSplit[hostNameSplit.length - 1];
        }
    }
    if (!(hostName in urlList)) {
        urlList[hostName] = []
    }
    urlList[hostName].push(addUrl);
}

for (var i = 0; i < seedList.length; i++) {
    addNewURL(seedList[i]);
}

app.post('/add_url', (req, res) => {
    // filter based on robots file, visited set, and blocked sites
    const addUrl = normalizeUrl(req.body.url, { stripHash: true })
    var passUrl = false;

    const newUrl = new URL(addUrl);

    if (!(newUrl.hostname in urlList)) {
        passUrl = true;
    } else {
        const r = visited.get(addUrl)
	console.log(r);
        if (r !== null && r.length > 0) {
            var addProb = 1 - r[0];
            if (addProb == 0) {
                addProb = 1 - r[1];
            }
            if (Math.random() < addProb) {
                passUrl = true;
            }
        } else {
            passUrl = true;
        }
    }
    if (passUrl) {
        addNewURL(addUrl);
    }
    res.sendStatus(200);
});

app.get('/request_url', (req, res) => {
    const domains = Object.keys(urlList);
    console.log(domains.length)
    if (domains.length == 0) {
        // console.log("ret null")
        res.json({ url: null });
    } else {
        // choose a random host name
        const randomDomain = domains[Math.floor(Math.random() * domains.length)];
        const retVal = urlList[randomDomain].shift()
        if (urlList[randomDomain].length == 0) {
            delete urlList[randomDomain];
        }
        visited.add(retVal);
        res.json({ url: retVal });
    }
});

app.get('/save_visited', (req, res) => {
    let visitedArray = visited.values()
    fs.writeFileSync('visited.json', JSON.stringify(visitedArray))
});

app.get('/save_urls', (req, res) => {
    fs.writeFileSync('urls.json', JSON.stringify(urlList));
});

app.listen(8090, '0.0.0.0', () => console.log('Started server!'));
