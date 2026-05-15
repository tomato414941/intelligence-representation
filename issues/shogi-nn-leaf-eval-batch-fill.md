# Shogi NN Leaf Eval Batch Fill

Status: open
Priority: medium

## Problem

Shogi generated experience can configure an NN leaf eval batch limit, but actual
batches may not fill close to that limit.

In the 2026-05-15 online replay run:

- configured NN leaf eval batch limit: 32
- self-play actual average batch size: about 6.1-6.4
- self-play fill ratio: about 0.19-0.20
- checkpoint-vs-USI actual average batch size: about 14.3-14.5
- checkpoint-vs-USI fill ratio: about 0.45

This means increasing the configured limit may not improve GPU utilization if
the CPU/search side cannot supply enough leaf positions.

## Desired Shape

Generated-experience measurements should keep configured batch limits separate
from actual batch fill.

When tuning generation throughput, the project should use:

- configured NN leaf eval batch limit
- actual NN leaf eval batch size average and max
- fill ratio
- source type
- concurrent games and worker processes

This issue is measurement and interpretation first. It should not force a search
algorithm rewrite before the bottleneck is clear.

## Close Condition

- Current generated-experience summaries expose actual NN leaf eval batch fill.
- Throughput docs record actual fill when comparing settings.
- A follow-up optimization decision can be made from measured fill data rather
  than configured batch size alone.
