package server

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
)

func TestSchedCancelledWaiterPreservesExplicitUnload(t *testing.T) {
	for _, order := range []string{"before_slot_claim", "after_slot_claim"} {
		t.Run(order, func(t *testing.T) {
			testCancelledWaiterPreservesEviction(t, order == "before_slot_claim", func(s *Scheduler, m *Model) { s.expireRunner(m) })
		})
	}
}

func TestSchedCancelledWaiterPreservesRuntimeOOMEviction(t *testing.T) {
	for _, order := range []string{"before_slot_claim", "after_slot_claim"} {
		t.Run(order, func(t *testing.T) {
			testCancelledWaiterPreservesEviction(t, order == "before_slot_claim", func(s *Scheduler, m *Model) {
				s.expireRunnersForRuntimeOOM(m, errors.New("cudaMalloc failed: out of memory"))
			})
		})
	}
}

func testCancelledWaiterPreservesEviction(t *testing.T, evictBefore bool, evict func(*Scheduler, *Model)) {
	t.Helper()
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", "1")
	s := InitScheduler(ctx)
	m := &Model{Name: "resident", ModelPath: "resident"}
	r := withModel(&runnerRef{
		llama:           &mockLlm{},
		modelKey:        "resident",
		refCount:        1,
		sessionDuration: time.Hour,
	}, m)
	s.loaded["resident"] = r
	if evictBefore {
		evict(s, m)
		r.expiresAt = r.expiresAt.Add(-time.Hour)
	}
	initialExpiry := r.expiresAt

	waiterCtx, cancelWaiter := context.WithCancel(ctx)
	defer cancelWaiter()
	opts := api.DefaultOptions()
	opts.NumGPU = 0
	s.getRunner(waiterCtx, &Model{Name: "queued", ModelPath: "queued"}, opts, nil, false, false, nil)
	var loops sync.WaitGroup
	loops.Add(1)
	go func() { defer loops.Done(); s.processPending(ctx) }()
	stopped := make(chan struct{})
	t.Cleanup(func() {
		cancel()
		go func() { loops.Wait(); close(stopped) }()
		select {
		case <-stopped:
		case <-time.After(2 * time.Second):
			t.Error("completed loop did not stop")
			return
		}
		r.refMu.Lock()
		defer r.refMu.Unlock()
		if r.expireTimer != nil {
			r.expireTimer.Stop()
		}
	})
	require.Eventually(t, func() bool {
		r.refMu.Lock()
		defer r.refMu.Unlock()
		return r.state == runnerStateEvictRequested && r.expiresAt.After(initialExpiry)
	}, 2*time.Second, time.Millisecond)
	if !evictBefore {
		evict(s, m)
	}
	cancelWaiter()
	s.pendingReqCh <- &LlmRequest{ctx: waiterCtx, model: &Model{Name: "barrier"}}
	require.Eventually(t, func() bool { return len(s.pendingReqCh) == 0 }, 2*time.Second, time.Millisecond)
	loops.Add(1)
	go func() { defer loops.Done(); s.processCompleted(ctx) }()
	s.finishedReqCh <- &LlmRequest{ctx: ctx, model: m}
	if !assertResidentUnloaded(s, time.Second) {
		r.refMu.Lock()
		state, refs, timer := r.state, r.refCount, r.expireTimer != nil
		r.refMu.Unlock()
		t.Fatalf("independent eviction was canceled with unrelated waiter: state=%v refs=%d timer=%v", state, refs, timer)
	}
}

func assertResidentUnloaded(s *Scheduler, timeout time.Duration) bool {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	tick := time.NewTicker(time.Millisecond)
	defer tick.Stop()
	for {
		s.loadedMu.Lock()
		_, loaded := s.loaded["resident"]
		s.loadedMu.Unlock()
		if !loaded {
			return true
		}
		select {
		case <-timer.C:
			return false
		case <-tick.C:
		}
	}
}
