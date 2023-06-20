#!/bin/bash

module load fsl

which fsl

FSLDIR=/panfs/roc/msisoft/fsl
. ${FSLDIR}/6.0.4/etc/fslconf/fsl.sh
PATH=${FSLDIR}/6.0.4/bin:${PATH}
export FSLDIR PATH