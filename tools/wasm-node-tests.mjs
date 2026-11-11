// Runs the onnxrt-kt wasmJs tests in Node with onnxruntime-web, so the web implementation is tested
// with real inference. (`./kotlin test -p wasmJs` runs them in a browser without onnxruntime-web.)
//
//   npm install --prefix build/ortweb onnxruntime-web@1.31.1
//   ./kotlin test -m onnxrt-kt -p wasmJs      # builds the test bundle
//   node --experimental-wasm-imported-strings tools/wasm-node-tests.mjs \
//        build/tasks/_onnxrt-kt_buildWasmJsAppWasmJsTestDebug \
//        build/ortweb/node_modules/onnxruntime-web/dist/ort.node.min.mjs
//
// Exits with status 1 if any test fails.
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const [bundleDir, ortModule] = process.argv.slice(2);
if (!bundleDir || !ortModule) {
    console.error('Usage: node --experimental-wasm-imported-strings tools/wasm-node-tests.mjs <test bundle dir> <ort.node.min.mjs>');
    process.exit(2);
}
process.env.ORT_WEB_MODULE = resolve(ortModule);

let failed = 0, finished = 0, openSuites = 0, lastMessageAt = performance.now();
const log = console.log.bind(console);
console.log = (...args) => {
    const line = args.map(String).join(' ');
    if (line.startsWith('##teamcity[testSuiteStarted')) openSuites++;
    else if (line.startsWith('##teamcity[testSuiteFinished')) openSuites--;
    else if (line.startsWith('##teamcity[testFailed')) { failed++; log('FAILED ' + (line.match(/name='([^']*)'/)?.[1] ?? '') + ': ' + (line.match(/message='([^']*)'/)?.[1] ?? '')); }
    else if (line.startsWith('##teamcity[testFinished')) { finished++; log('done   ' + (line.match(/name='([^']*)'/)?.[1] ?? '')); }
    else if (!line.startsWith('##teamcity')) log(line);
    lastMessageAt = performance.now();
};

const testModule = await import(pathToFileURL(resolve(bundleDir, 'onnxrt-kt_test.mjs')).href);
testModule['startUnitTests']?.();
while (!(finished > 0 && openSuites === 0 && performance.now() - lastMessageAt > 500)) {
    await new Promise(r => setTimeout(r, 50));
}
log(`${finished} tests finished, ${failed} failed`);
process.exit(failed ? 1 : 0);
