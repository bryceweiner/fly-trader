# fly-trader rules

## Live money first and fast over paper, always

When live money is on the line, live orders execute first and fast; the pipeline makes way for them and every paper
trade comes after. In each minute the live exits go out at the minute boundary (they depend on nothing but time) and
the live entries the moment the fly has decided, before any other book's scoring, paper fill, learning, mark, state
or status write. No change may put paper, learning or bookkeeping work in front of a live order; check live order
latency first when designing or reviewing any change to the minute pipeline.
