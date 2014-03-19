#!/bin/sh

test_description='Test FE interpolation'

. ./fefas-sharness.sh

test_expect_stdout 'FE Interpolation fedegree=1 serial' 1 'fefas test-feinterp -M 6,2,10 -L 6,2,10' '
|u - I Ihat u|_max =     0
'

test_expect_stdout 'FE Interpolation fedegree=1 parallel' 4 'fefas test-feinterp -M 6,2,10 -L 6,2,10 -p 2,1,2' '
|u - I Ihat u|_max =     0
'

test_done
