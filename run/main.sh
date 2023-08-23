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
# where the x-y points in the plots actually used in paper will be put
XYDir=${myDir}/../plotXYData/
echo 'paperDir is '${XYDir}
mkdir -p -v ${XYDir}
# rm -v ${XYDir}/*
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
    unzip ${T}x32.zip
done
cd -
echo ''
# The correlators will now be in
# correlators/correlators/${T}x32/data/{op}_${qqq}/

# The parity ratio plots are quick - ~3 min
echo ${pythonExe} ${scriptDir}/parityRatios.py ${myDir}/completeToml/parityJ1_2.toml
echo 'date is '$(date)
${pythonExe} ${scriptDir}/parityRatios.py ${myDir}/completeToml/parityJ1_2.toml
echo 'date is '$(date)
# COPY
echo cp -av ${plotDir}/parityRatio/SummedParityRatio.pdf ${paperDir}/SummedParityRatio.pdf
cp -av ${plotDir}/parityRatio/SummedParityRatio.pdf ${paperDir}/SummedParityRatio.pdf
cp -av ${plotDir}/parityRatio/SummedParityRatio_Page0.csv ${XYDir}/SummedParityRatio.csv
# Generate the ratio plots - ~<30 min
# Plots of the correlator to the model correlator
# A script for the single
# and a script for the double
# and for different temps
# for y-limit purposes
echo bash ${myDir}/ModelRatios.sh
echo 'date is '$(date)
bash ${myDir}/ModelRatios.sh
echo bash ${myDir}/ModelRatios_mid.sh
echo 'date is '$(date)
bash ${myDir}/ModelRatios_mid.sh
echo 'date is '$(date)
echo bash ${myDir}/DoubleModelRatios.sh
echo 'date is '$(date)
bash ${myDir}/DoubleModelRatios.sh
echo 'date is '$(date)
# and the 'mid temperature
echo bash ${myDir}/DoubleModelRatios_mid.sh
echo 'date is '$(date)
bash ${myDir}/DoubleModelRatios_mid.sh
echo 'date is '$(date)
# COPY PLOTS
for temp in cool mid; do
    for OP in sigma12_3fl_udc doublet_2fl_ccs doublet_2fl_ccu; do
	for RAT in double single; do
	    for PAR in Pos Neg; do
		echo ''
		# echo ${plotDir}/ratios/12_06_2023/${OP}/${temp}/${RAT}/${PAR}/G_ModelRatio.pdf
		# echo ${paperDir}/MR_${OP}_${temp}_${RAT}_${PAR}.pdf
		cp -av ${plotDir}/ratios/12_06_2023/${OP}/${temp}/${RAT}/${PAR}/G_ModelRatio.pdf ${paperDir}/MR_${OP}_${temp}_${RAT}_${PAR}.pdf
		cp -av ${plotDir}/ratios/12_06_2023/${OP}/${temp}/${RAT}/${PAR}/G_ModelRatio_Page0.csv ${XYDir}/MR_${OP}_${temp}_${RAT}_${PAR}.csv
	    done
	done
    done
done
#cp -av ${plotDir}/

# Uncomment below to do plots of the model correlator itself
# echo bash ${myDir}/ModelCorr.sh
# bash ${myDir}/ModelCorr.sh

# Plots of the correlator to the recon correlator
echo bash ${myDir}/ReconRatios.sh
echo 'date is '$(date)
bash ${myDir}/ReconRatios.sh
echo 'date is '$(date)
#and the mid temperature
echo bash ${myDir}/ReconRatios_mid.sh
echo 'date is '$(date)
bash ${myDir}/ReconRatios_mid.sh
echo 'date is '$(date)
# COPY PLOTS
for temp in cool mid; do
    for OP in sigma12_3fl_udc doublet_2fl_ccs; do
	for RAT in single; do
	    for PAR in Pos Neg; do
		echo ''
		# echo ${plotDir}/ratios/12_06_2023/${OP}/${temp}/${RAT}/${PAR}/G_ReconRatio.pdf
		# echo ${paperDir}/RR_${OP}_${temp}_${RAT}_${PAR}.pdf
		cp -av ${plotDir}/ratios/12_06_2023/${OP}/${temp}/${RAT}/${PAR}/G_ReconRatio.pdf ${paperDir}/RR_${OP}_${temp}_${RAT}_${PAR}.pdf
		cp -av ${plotDir}/ratios/12_06_2023/${OP}/${temp}/${RAT}/${PAR}/G_ReconRatio_Page0.csv ${paperDir}/RR_${OP}_${temp}_${RAT}_${PAR}.csv
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
echo 'date is '$(date)
bash ${myDir}/run_fits.sh
echo 'date is '$(date)

# Now copy the specific plots for the xi_cc 128
xicc=${plotDir}/fits/doublet_2fl_ccu_128x32/mAve.pdf
xicc_csv=${plotDir}/fits/doublet_2fl_ccu_128x32/mAve
pdfseparate -f 1 -l 1 ${xicc} ${paperDir}/Figure1.pdf
pdfseparate -f 14 -l 14 ${xicc} ${paperDir}/Figure2.pdf
pdfseparate -f 18 -l 18 ${xicc} ${paperDir}/Figure3.pdf
pdfseparate -f 19 -l 19 ${xicc} ${paperDir}/Figure4.pdf
# copy the csvs
cp -av ${xicc_csv}_Page0.csv ${XYDir}/Figure1.csv
cp -av ${xicc_csv}_Page13.csv ${XYDir}/Figure2.csv
cp -av ${xicc_csv}_Page17.csv ${XYDir}/Figure3.csv
cp -av ${xicc_csv}_Page18.csv ${XYDir}/Figure4.csv


echo 'date is '$(date)
# Run the code to add the systematic error from the choice of averaging method
echo ${pythonExe} ${scriptDir}/addSysErr.py ${myDir}/completeToml/addSysErr.toml
${pythonExe} ${scriptDir}/addSysErr.py ${myDir}/completeToml/addSysErr.toml


# Now the spectrum plot
echo ${pythonExe} ${scriptDir}/plotSpectrum.py ${myDir}/completeToml/plotSpectrumJ1_2.toml
${pythonExe} ${scriptDir}/plotSpectrum.py ${myDir}/completeToml/plotSpectrumJ1_2.toml
cp -av ${plotDir}/spectrumPlot.pdf ${paperDir}/spectrumPlot.pdf
cp -av ${plotDir}/spectrumPlot_Page0.csv ${XYDir}/spectrumPlot.csv

# now the mass as a function of temperature plot
# C = 1
echo ${pythonExe} ${scriptDir}/singlePlotSepNorm.py ${myDir}/completeToml/singleJ1_2_C1.toml
${pythonExe} ${scriptDir}/singlePlotSepNorm.py ${myDir}/completeToml/singleJ1_2_C1.toml
cp -av ${plotDir}/C1/singlePlotSepNorm_BothParity.pdf ${paperDir}/C1_BothParity.pdf
cp -av ${plotDir}/C1/singlePlotSepNorm_BothParity_Page0.csv ${XYDir}/C1_BothParity.csv
# C = 2
echo ${pythonExe} ${scriptDir}/singlePlotSepNorm.py ${myDir}/completeToml/singleJ1_2_C2.toml
${pythonExe} ${scriptDir}/singlePlotSepNorm.py ${myDir}/completeToml/singleJ1_2_C2.toml
cp -av ${plotDir}/C2/singlePlotSepNorm_BothParity.pdf ${paperDir}/C2_BothParity.pdf
cp -av ${plotDir}/C2/singlePlotSepNorm_BothParity_Page0.csv ${XYDir}/C2_BothParity.csv
echo 'date is '$(date)
