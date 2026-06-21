const assert = require('node:assert/strict')
const test = require('node:test')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const {
  runBootstrap,
  resolveInstallScript,
  installedAgentInstallScript,
  cachedScriptPath
} = require('./bootstrap-runner.cjs')

const SCRIPT_NAME = process.platform === 'win32' ? 'install.ps1' : 'install.sh'

function mkTmpHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-test-'))
}

test('runBootstrap bails immediately when the signal is already aborted', async () => {
  const controller = new AbortController()
  controller.abort()

  const events = []
  const result = await runBootstrap({
    installStamp: null,
    activeRoot: '/tmp/hermes-runner-test',
    sourceRepoRoot: null,
    hermesHome: '/tmp/hermes-runner-test',
    logRoot: '/tmp/hermes-runner-test',
    onEvent: ev => events.push(ev),
    abortSignal: controller.signal
  })

  // Cancelled before any install script is spawned.
  assert.deepEqual(result, { ok: false, cancelled: true })
  assert.ok(
    events.some(ev => ev.type === 'failed' && /cancelled/i.test(ev.error)),
    'should emit a cancelled failure event'
  )
})

test('installedAgentInstallScript resolves the installer in the agent checkout', () => {
  const home = mkTmpHome()
  try {
    assert.equal(installedAgentInstallScript(home), null, 'absent before the checkout exists')

    const scriptsDir = path.join(home, 'hermes-agent', 'scripts')
    fs.mkdirSync(scriptsDir, { recursive: true })
    const scriptPath = path.join(scriptsDir, SCRIPT_NAME)
    fs.writeFileSync(scriptPath, '#!/bin/sh\necho hi\n')

    assert.equal(installedAgentInstallScript(home), scriptPath)
    assert.equal(installedAgentInstallScript(null), null, 'null home -> null')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript prefers a cached script without touching the network', async () => {
  const home = mkTmpHome()
  try {
    const commit = 'a'.repeat(40)
    const cached = cachedScriptPath(home, commit)
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, '#!/bin/sh\necho cached\n')

    const logs = []
    const result = await resolveInstallScript({
      installStamp: { commit },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: ev => logs.push(ev)
    })

    assert.equal(result.source, 'cache')
    assert.equal(result.path, cached)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript falls back to the installed agent checkout on a 404', async () => {
  const home = mkTmpHome()
  try {
    const commit = 'a'.repeat(40)
    // Seed the installed agent checkout so the fallback has something to resolve.
    const scriptsDir = path.join(home, 'hermes-agent', 'scripts')
    fs.mkdirSync(scriptsDir, { recursive: true })
    const installed = path.join(scriptsDir, SCRIPT_NAME)
    fs.writeFileSync(installed, '#!/bin/sh\necho fallback\n')

    const logs = []
    const result = await resolveInstallScript({
      installStamp: { commit },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: ev => logs.push(ev),
      // Simulate GitHub returning a 404 for the pinned commit.
      _download: async () => {
        throw new Error('Failed to download install.sh: HTTP 404')
      }
    })

    assert.equal(result.source, 'installed-agent')
    // It should have copied the installer into the bootstrap cache.
    assert.equal(result.path, cachedScriptPath(home, commit))
    assert.ok(fs.existsSync(result.path), 'fallback script copied into cache')
    assert.ok(
      logs.some(ev => /falling back to installed agent/.test(ev.line || '')),
      'emits a fallback log line'
    )
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript rethrows when the 404 fallback is unavailable', async () => {
  const home = mkTmpHome()
  try {
    const commit = 'a'.repeat(40)
    // No installed agent checkout seeded -> nothing to fall back to.
    await assert.rejects(
      resolveInstallScript({
        installStamp: { commit },
        sourceRepoRoot: null,
        hermesHome: home,
        emit: () => {},
        _download: async () => {
          throw new Error('Failed to download install.sh: HTTP 404')
        }
      }),
      /HTTP 404|Failed to download/
    )
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

// Regression tests for #49982 — confirm the install.sh contract that
// bootstrap-runner.cjs depends on. The Electron desktop app calls
// install.sh --manifest to discover the stage list, then runs each
// stage via install.sh --json --stage <name> <args>. Both modes must
// emit parseable JSON on stdout so the renderer's install overlay can
// stream progress without the user staring at a generic error.
//
// These tests exercise the real scripts/install.sh in this repository
// (the same file bootstrap-runner would resolve at runtime), and are
// Linux/macOS only — Windows uses install.ps1 and a parallel test
// there will land with the Windows bootstrap path.
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const LOCAL_INSTALL_SH = path.join(REPO_ROOT, 'scripts', 'install.sh')

function skipIfNoInstallScript() {
  if (process.platform === 'win32') {
    // Windows uses install.ps1; bootstrap-runner's Posix/PowerShell split
    // is exercised by bootstrap-platform.test.cjs.
    return 'install.ps1 contract test runs only on Posix'
  }
  if (!fs.existsSync(LOCAL_INSTALL_SH)) {
    return `scripts/install.sh not present at ${LOCAL_INSTALL_SH}`
  }
  return null
}

test('install.sh --manifest emits a parseable stage contract (Phase 1D)', t => {
  const skip = skipIfNoInstallScript()
  if (skip) {
    t.skip(skip)
    return
  }

  const { execFileSync } = require('node:child_process')
  const stdout = execFileSync(LOCAL_INSTALL_SH, ['--manifest'], {
    encoding: 'utf8'
  })
  const manifest = JSON.parse(stdout)

  assert.equal(manifest.protocol_version, 1, 'manifest protocol_version=1')
  assert.ok(Array.isArray(manifest.stages), 'manifest.stages must be an array')
  assert.ok(manifest.stages.length > 0, 'manifest.stages must not be empty')

  // Pin a few stage names that bootstrap-runner.cjs and the renderer
  // overlay depend on. Adding a new stage is fine, renaming or
  // removing these is a breaking change for the desktop installer.
  const stageNames = manifest.stages.map(s => s.name)
  for (const required of [
    'prerequisites',
    'repository',
    'venv',
    'python-deps',
    'complete'
  ]) {
    assert.ok(
      stageNames.includes(required),
      `manifest must declare stage '${required}' (got: ${stageNames.join(', ')})`
    )
  }

  // Every stage must declare a category — the renderer uses it to
  // group progress rows.
  for (const stage of manifest.stages) {
    assert.ok(
      typeof stage.category === 'string' && stage.category.length > 0,
      `stage ${stage.name} missing category`
    )
    assert.equal(
      typeof stage.needs_user_input,
      'boolean',
      `stage ${stage.name} needs_user_input must be boolean`
    )
  }
})

test('install.sh --json --stage prerequisites runs cleanly when prereqs are met', t => {
  const skip = skipIfNoInstallScript()
  if (skip) {
    t.skip(skip)
    return
  }

  const { execFileSync } = require('node:child_process')
  // The prerequisites stage is the cheap one (no network, no install).
  // On a developer machine git/node/internet are already present, so it
  // returns {"ok":true}. If the runner is invoked on a host without
  // them, JSONDecodeError must NOT mask the failure — the JSON line
  // is still emitted with ok=false.
  let stdout
  try {
    stdout = execFileSync(LOCAL_INSTALL_SH, ['--json', '--stage', 'prerequisites'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    })
  } catch (err) {
    stdout = err.stdout ? err.stdout.toString() : ''
    // Non-zero exit is fine here — what we care about is that we can
    // parse a JSON frame out of stdout regardless.
  }
  const line = stdout
    .split('\n')
    .reverse()
    .find(l => l.trim().startsWith('{'))
  assert.ok(line, `expected at least one JSON line in: ${stdout}`)
  const frame = JSON.parse(line)
  assert.equal(frame.stage, 'prerequisites')
  assert.equal(typeof frame.ok, 'boolean')
})
