FROM golang:1.26.8-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/lil-fleet ./cmd/lil-fleet

FROM alpine:3.23
RUN apk add --no-cache ca-certificates docker-cli
COPY --from=build /out/lil-fleet /usr/local/bin/lil-fleet
ENTRYPOINT ["lil-fleet"]
CMD ["-manifest", "/etc/lil-fleet/fleet.json"]
