package deployment

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestSanitizeMessageRedactsStripsAndCaps(t *testing.T) {
	got := SanitizeMessage("token=sk-live-123456 ok=1\x1b[2J\r\nnext\tline", []string{"sk-live-123456", "1", ""})
	if got != "token=[redacted] ok=1 [2J next line" {
		t.Fatalf("SanitizeMessage = %q", got)
	}
	long := SanitizeMessage(strings.Repeat("é", MaxMessageBytes), nil)
	if len(long) > MaxMessageBytes || !utf8.ValidString(long) || !strings.HasSuffix(long, "...") {
		t.Fatalf("capped message length=%d valid=%v", len(long), utf8.ValidString(long))
	}
}
