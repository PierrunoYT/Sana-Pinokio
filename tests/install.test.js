const assert = require('node:assert/strict')
const test = require('node:test')
const vm = require('node:vm')
const install = require('../install.js')
const torch = require('../torch.js')
const start = require('../start.js')
const update = require('../update.js')

function evaluate(value, context) {
  return typeof value === 'string' && value.startsWith('{{')
    ? vm.runInNewContext(value.slice(2, -2), context)
    : value
}

// Expand only the installer's shell/script steps, not external commands. This
// checks the input -> local -> child args handoff as well as each device branch.
function installSteps(device, platform, gpu) {
  const context = { input: { device }, local: {}, platform, gpu, exists: () => false }
  const steps = []
  for (const step of install.run) {
    if (step.when && !evaluate(step.when, context)) continue
    if (step.method === 'local.set') {
      for (const [key, value] of Object.entries(step.params)) {
        context.local[key] = evaluate(value, context)
      }
    } else if (step.method === 'script.start') {
      assert.equal(step.params.uri, 'torch.js')
      const childContext = { ...context, args: step.params.params }
      const selected = torch.run.filter(child => evaluate(child.when, childContext))
      assert.equal(selected.length, 1)
      for (const child of selected) {
        steps.push({ ...child, params: Object.fromEntries(
          Object.entries(child.params).map(([key, value]) => [key, evaluate(value, childContext)])
        ) })
      }
    } else {
      steps.push(step)
    }
  }
  return steps
}

const devices = [
  ['win32', 'nvidia'], ['win32', 'amd'], ['win32', undefined],
  ['linux', 'nvidia'], ['linux', 'amd'], ['linux', undefined],
  ['darwin', 'apple'],
]

for (const device of ['cpu', 'gpu']) {
  for (const [platform, gpu] of devices) {
    test(`${device}/${platform}/${gpu ?? 'none'} routes install and launch commands to one environment`, () => {
      const steps = installSteps(device, platform, gpu)
      const shells = [
        ...steps.filter(step => step.method === 'shell.run'),
        ...start.run.filter(step => step.method === 'shell.run'),
        ...update.run.filter(step => step.method === 'shell.run' && step.params.venv),
      ]
      for (const step of shells) {
        assert.equal(step.params.path, '.', 'Python commands must run at the project root')
        assert.equal(step.params.venv, 'env', 'Python commands must activate the launch environment')
      }
      const messages = steps.map(step => [step.params?.message].flat().join('\n'))
      const torchIndex = messages.findIndex(message => message.includes('uv pip install') && message.includes('torch=='))
      const requirementsIndex = messages.findIndex(message => message.includes('uv pip install -r app/requirements.txt'))
      const verificationIndex = messages.findIndex(message => message.includes('import torch'))
      const downloadIndex = steps.findIndex(step => step.method === 'hf.download')
      assert.ok(torchIndex >= 0, 'every installation choice must install PyTorch')
      assert.ok(torchIndex < requirementsIndex, 'install PyTorch before app dependencies')
      assert.ok(requirementsIndex < verificationIndex, 'verify imports after installing dependencies')
      assert.ok(verificationIndex < downloadIndex, 'detect missing PyTorch before downloading the model')
    })
  }
}
