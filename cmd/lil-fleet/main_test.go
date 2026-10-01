package main

import (
	"net/http"
	"testing"
)

func TestUnauthenticatedAPIRequiresLoopbackOrExplicitOptOut(t *testing.T) {
	for _, tc := range []struct {
		listen   string
		token    bool
		insecure bool
		wantErr  bool
	}{
		{"127.0.0.1:8090", false, false, false},
		{"[::1]:8090", false, false, false},
		{"localhost:8090", false, false, false},
		{"0.0.0.0:8090", false, false, true},
		{":8090", false, false, true},
		{"[::]:8090", false, false, true},
		{"192.168.1.5:8090", false, false, true},
		{"0.0.0.0:8090", true, false, false},
		{"0.0.0.0:8090", false, true, false},
	} {
		err := checkListenAuthentication(tc.listen, tc.token, tc.insecure)
		if (err != nil) != tc.wantErr {
			t.Fatalf("checkListenAuthentication(%q, token=%v, insecure=%v) = %v", tc.listen, tc.token, tc.insecure, err)
		}
	}
}

func TestServerBoundsEveryConnectionPhase(t *testing.T) {
	server := newServer("127.0.0.1:0", http.NotFoundHandler())
	if server.ReadHeaderTimeout == 0 || server.ReadTimeout == 0 || server.WriteTimeout == 0 || server.IdleTimeout == 0 {
		t.Fatalf("server timeouts = %+v", server)
	}
}
