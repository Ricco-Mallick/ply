// The two rules the API now enforces on top of the gate: what an invite may spend, and when results are public.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { inviteBudget, resultsVisible, INVITE_ACTIVE_LIMIT, INVITE_DAILY_LIMIT } from '../src/limits.ts';

test('an invite gets one job in flight', () => {
  assert.equal(inviteBudget(0, 0), null);
  assert.match(String(inviteBudget(1, 3)), /1 job in flight/);
  assert.match(String(inviteBudget(2, 3)), /still/);
});

test('an invite has a daily ceiling and the team key has none before it', () => {
  assert.equal(inviteBudget(0, INVITE_DAILY_LIMIT - 1), null);
  assert.match(String(inviteBudget(0, INVITE_DAILY_LIMIT)), /a day/);
  assert.ok(INVITE_ACTIVE_LIMIT >= 1 && INVITE_DAILY_LIMIT > INVITE_ACTIVE_LIMIT);
});

test('results are public for a reviewed or approved twin, and for a legacy scene with no job row', () => {
  assert.equal(resultsVisible('review', 'scene.spz', true), true);
  assert.equal(resultsVisible('done', 'scene_lite.spz', true), true);
  assert.equal(resultsVisible(undefined, 'scene.spz', true), true);
});

test('a running job serves only what the live view needs', () => {
  assert.equal(resultsVisible('running', 'scene.spz', true), false);
  assert.equal(resultsVisible('running', 'live-index.json', true), true);
  assert.equal(resultsVisible('running', 'live/0042.spz', true), true);
  assert.equal(resultsVisible('running', 'pipeline.log', true), true);
});

test('a worker is never gated, and a failed job is not public', () => {
  assert.equal(resultsVisible('running', 'scene.spz', false), true);
  assert.equal(resultsVisible('failed', 'scene.spz', true), false);
});
