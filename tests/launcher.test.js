const assert = require('node:assert/strict')
const test = require('node:test')
const vm = require('node:vm')
const launcher = require('../pinokio.js')

function evaluate(template, context) {
  return vm.runInNewContext(template.slice(2, -2), context)
}

test('partial installations offer Install; ready installations offer Start', async () => {
  for (const ready of [false, true]) {
    const menu = await launcher.menu({}, {
      exists: path => path === 'env' || (ready && path === 'env/.installed'),
      running: () => false,
    })
    assert.equal(menu[0].href, ready ? 'start.js' : 'install.js')
    assert.equal(menu[0].default, true)
  }
})

test('maintenance remains visible after removing installation files', async () => {
  for (const script of ['install', 'update', 'reset', 'link']) {
    const menu = await launcher.menu({}, {
      exists: () => false,
      running: path => path === `${script}.js`,
    })
    assert.equal(menu[0].href, `${script}.js`)
  }
})

test('server URL capture opens the actual local endpoint', async () => {
  const start = require('../start.js')
  const pattern = start.run[0].params.on[0].event
  const event = new RegExp(pattern.slice(1, -1)).exec('Running on local URL:  http://127.0.0.1:7861')
  const url = evaluate(start.run[1].params.url, { input: { event } })
  assert.equal(url, 'http://127.0.0.1:7861')
  const menu = await launcher.menu({}, {
    exists: () => true, running: path => path === 'start.js', local: () => ({ url }),
  })
  assert.equal(menu[0].href, url)
})

test('each supported device selects exactly one matching torch installation', () => {
  const script = require('../torch.js')
  for (const [platform, gpu, index] of [
    ['win32', 'nvidia', 'cu128'], ['linux', 'nvidia', 'cu128'],
    ['win32', 'amd', 'cpu'], ['linux', 'amd', 'rocm6.3'],
    ['darwin', 'apple', 'cpu'], ['linux', undefined, 'cpu'], ['win32', undefined, 'cpu'],
  ]) {
    const steps = script.run.filter(step => evaluate(step.when, { platform, gpu }))
    assert.equal(steps.length, 1)
    const messages = [steps[0].params.message].flat().join(' ')
    assert.ok(messages.includes(`https://download.pytorch.org/whl/${index}`))
    assert.ok(messages.includes('--reinstall'))
    assert.ok(!messages.includes('torch-directml'))
  }
})

test('installation publishes readiness only after verification and model download', () => {
  const steps = require('../install.js').run
  const ready = steps.findIndex(step => step.method === 'fs.write' && step.params.path === 'env/.installed')
  const download = steps.findIndex(step => step.method === 'hf.download')
  const check = steps.findIndex(step => JSON.stringify(step.params).includes('uv pip check'))
  assert.ok(check < download && download < ready)
  const config = steps.find(step => step.method === 'fs.write' && step.params.json)
  assert.equal(evaluate(config.params.json.device, { local: { device: 'cpu' } }), 'cpu')
})
