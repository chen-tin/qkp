#!/bin/bash
# Detached re-run: two streams in parallel, then a final pass that adds the
# newly trained decoders (cached results are reused) and runs E13.
cd "$(dirname "$0")/../.."
export MPLBACKEND=Agg
PY=.venv/bin/python
L=results/logs
stamp() { echo "[$(date +%H:%M:%S)] $*" >> $L/rerun_driver.log; }
stamp "start (pid $$)"
( $PY -m experiments.run_nonstationary > $L/rerun_E9-E11.log 2>&1; stamp "E9-E11 exit $?";
  $PY -m experiments.run_all --only E12 > $L/rerun_E12.log 2>&1; stamp "E12 exit $?" ) &
( $PY -m training.train_everything > $L/rerun_train.log 2>&1; stamp "training exit $?" ) &
wait
stamp "streams done; final pass"
$PY -m experiments.run_all --only E9 E10 E11 E12 E13 E14 E15 > $L/rerun_final.log 2>&1
stamp "final pass exit $?"
$PY -m experiments.analyze_nonstationary > $L/rerun_analyze.log 2>&1
stamp "analysis exit $?; all done"
