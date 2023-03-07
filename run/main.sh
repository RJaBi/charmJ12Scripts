#!/bin/bash
#Where we are running this from
# myDir='/home/ryan/Documents/GitHub/PullRequests_SwanLat/cfuns/tomlWriters/charmBaryon/'
# This line spits a 'Bad Substition' error but works
myDir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo 'myDir is '${myDir}
# where the scripts are
scriptDir=${myDir}/../code/bin/
echo 'scriptDir is '${scriptDir}
# The python executable
pythonExe=python
echo 'pythonExe is '${pythonExe}
# where the data is kept
cfDir=${myDir}/../correlators/
echo 'cfDir is '${cfDir}
# where the plots actually used in paper will be put
paperDir=${myDir}/../paperPlots/
echo 'paperDir is '${paperDir}
mkdir -p -v ${paperDir}
rm -v ${paperDir}/*
# where the plots are generated to
plotDir=${myDir}/../output/
echo 'plotDir is '${plotDir}
echo ''

# The first step is to unzip all the correlators
mkdir -p -v ${cfDir}
cd ${cfDir}

for T in 16 20 24 28 32 36 40 48 56 64 128
do
    echo unzip ${T}x32.zip
    #unzip ${T}x32.zip
done
cd -
echo ''
# The correlators will now be in
# correlators/correlators/${T}x32/data/{op}_${qqq}/

# The parity ratio plots are quick - ~3 min
echo ${pythonExe} ${scriptDir}/parityRatios.py ${myDir}/completeToml/parityJ1_2.toml
# ${pythonExe} ${scriptDir}/parityRatios.py ${myDir}/completeToml/parityJ1_2.toml
# COPY
echo cp -av ${plotDir}/parityRatio/SummedParityRatio.pdf ${paperDir}/SummedParityRatio.pdf
# cp -av ${plotDir}/parityRatio/SummedParityRatio.pdf ${paperDir}/SummedParityRatio.pdf
# and the inflection points plot
echo ${pythonExe} ${scriptDir}/plotInflectionPoints.py ${myDir}/completeToml/inflectionPoints.toml
# ${pythonExe} ${scriptDir}/plotInflectionPoints.py ${myDir}/completeToml/inflectionPoints.toml
# COPY
echo cp -av ${plotDir}/inflectionPoints.pdf ${paperDir}/inflectionPoints.pdf
# cp -av ${plotDir}/inflectionPoints.pdf ${paperDir}/inflectionPoints.pdf

# Generate the ratio plots - ~<30 min
# Plots of the correlator to the model correlator
# echo bash ${myDir}/ModelRatios.sh
#bash ${myDir}/ModelRatios.sh
# COPY PLOTS
for temp in cool mid; do
    for OP in sigma12_3fl_udc doublet_2fl_ccs; do
	for RAT in double single; do
	    for PAR in Pos Neg; do
		echo ''
		echo ${plotDir}/ratios/02_03_2023/${OP}/${temp}/${RAT}/${PAR}/G_ModelRatio.pdf
		echo ${paperDir}/MR_${OP}_${temp}_${RAT}_${PAR}.pdf
		# cp -av ${plotDir}/ratios/02_03_2023/${OP}/${temp}/${RAT}/${PAR}/G_ModelRatio.pdf ${paperDir}/MR_${OP}_${temp}_${RAT}_${PAR}.pdf
	    done
	done
    done
done
#cp -av ${plotDir}/

# Uncomment below to do plots of the model correlator itself
# echo bash ${myDir}/ModelCorr.sh
# bash ${myDir}/ModelCorr.sh

# Plots of the correlator to the recon correlator
# echo bash ${myDir}/ReconRatios.sh
# bash ${myDir}/ReconRatios.sh
# COPY PLOTS
for temp in cool mid; do
    for OP in sigma12_3fl_udc doublet_2fl_ccs; do
	for RAT in single; do
	    for PAR in Pos Neg; do
		echo ''
		echo ${plotDir}/ratios/02_03_2023/${OP}/${temp}/${RAT}/${PAR}/G_ReconRatio.pdf
		echo ${paperDir}/RR_${OP}_${temp}_${RAT}_${PAR}.pdf
		# cp -av ${plotDir}/ratios/02_03_2023/${OP}/${temp}/${RAT}/${PAR}/G_ReconRatio.pdf ${paperDir}/RR_${OP}_${temp}_${RAT}_${PAR}.pdf
	    done
	done
    done
done
# Uncomment below to do plots of the Recon correlator itself
# echo bash ${myDir}/ReconCorr.sh
# bash ${myDir}/ReconCorr.sh

# The fits take somewhat longer
# Still less than 1 hour.
echo bash ${myDir}/run_fits.sh
# bash ${myDir}/run_fits.sh

# Now copy the specific plots for the xi_cc 128
xicc=${plotDir}/fits/doublet_2fl_ccu_128x32/mAve.pdf
pdfseparate -f 1 -l 1 ${xicc} ${paperDir}/Figure1.pdf
pdfseparate -f 14 -l 14 ${xicc} ${paperDir}/Figure2.pdf
pdfseparate -f 17 -l 17 ${xicc} ${paperDir}/Figure3.pdf
pdfseparate -f 18 -l 18 ${xicc} ${paperDir}/Figure4.pdf

# Run the code to add the systematic error from the choice of averaging method
echo ${pythonExe} ${scriptDir}/addSysErr.py ${myDir}/completeToml/addSysErr.toml
# ${pythonExe} ${scriptDir}/addSysErr.py ${myDir}/completeToml/addSysErr.toml


# Now the spectrum plot
echo ${pythonExe} ${scriptDir}/plotSpectrum.py ${myDir}/completeToml/plotSpectrumJ1_2.toml
# ${pythonExe} ${scriptDir}/plotSpectrum.py ${myDir}/completeToml/plotSpectrumJ1_2.toml
# cp -av ${plotDir}/spectrumPlot.pdf ${paperDir}/spectrumPlot.pdf

