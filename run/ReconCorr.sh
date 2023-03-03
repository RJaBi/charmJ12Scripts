#!/bin/bash -xe

#Where we are running this from
# myDir='/home/ryan/Documents/GitHub/PullRequests_SwanLat/cfuns/tomlWriters/charmBaryon/'
myDir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# The python executable
pythonExe=python #/home/ryan/installs/conda/miniconda3/envs/gen/bin/python

# The script used to turn the template into the full files
writeScript=${myDir}/writeReconRatios.py

# Where we are putting all the modified toml files
outTomlDir=${myDir}/AutoToml_Ratios/02_03_2023/ReconCorr/
# Make a folder to put the modified toml in
mkdir -p ${outTomlDir}

# the plotting script
#scriptDir=/home/ryan/Documents/GitHub/PullRequests_SwanLat/cfuns/currentlyUsed/
scriptDir=${myDir}/../code/bin/
exe=GPlots.py

# Data Dir
#dataDir=/home/ryan/Documents/2022/Gen2L/Hawk/charmBaryonData/
#dataDir=/home/ryan/Documents/2023/Gen2L/charmBaryonDataJ1_2/
dataDir=${myDir}/../

# Loop over temperatures - cool, mid, hot
#for temp in 48;
for temp in cool mid;
#for temp in cool mid hot;
#for temp in cool;
do
    # The template toml file
    template=${myDir}/template-${temp}-base.toml
    
#Charmless#    # Nucleon
#Charmless#    OP=doublet.2fl
#Charmless#    UN=doublet_2fl
#Charmless#    qqq=uud
#Charmless#    had='N(uud)'
#Charmless#    for RAT in G;
#Charmless#    #for RAT in single;
#Charmless#    do
#Charmless#	PAR=Pos
#Charmless#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#Charmless#	echo cp -av ${template} ${thisToml}
#Charmless#	cp -av ${template} ${thisToml}
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#Charmless#     find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;

#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#Charmless#	# Run the code to generate the full file
#Charmless#	echo ${pythonExe} ${writeScript} ${thisToml}
#Charmless#	${pythonExe} ${writeScript} ${thisToml}
#Charmless#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#Charmless#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#Charmless#	cd ${dataDir}
#Charmless#	# and run the code to generate the plots
#Charmless#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	cd -
#Charmless#	# Negative parity
#Charmless#	PAR=Neg
#Charmless#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#Charmless#	echo cp -av ${template} ${thisToml}
#Charmless#	cp -av ${template} ${thisToml}
#Charmless#     find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#Charmless#	# Run the code to generate the full file
#Charmless#	echo ${pythonExe} ${writeScript} ${thisToml}
#Charmless#	${pythonExe} ${writeScript} ${thisToml}
#Charmless#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#Charmless#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#Charmless#	cd ${dataDir}
#Charmless#	# and run the code to generate the plots
#Charmless#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	cd -
#Charmless#    done
#Charmless#    
#Charmless#    # Sigma(uus)
#Charmless#    OP=sigma12.3fl
#Charmless#    UN=sigma12_3fl
#Charmless#    qqq=uds
#Charmless#    had='\Sigma(uus)'
#Charmless#    for RAT in G;
#Charmless#    do
#Charmless#	PAR=Pos
#Charmless#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#Charmless#	echo cp -av ${template} ${thisToml}
#Charmless#	cp -av ${template} ${thisToml}
#Charmless#     find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#Charmless#	# Run the code to generate the full file
#Charmless#	echo ${pythonExe} ${writeScript} ${thisToml}
#Charmless#	${pythonExe} ${writeScript} ${thisToml}
#Charmless#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#Charmless#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#Charmless#	cd ${dataDir}
#Charmless#	# and run the code to generate the plots
#Charmless#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	cd -
#Charmless#	# Negative parity
#Charmless#	PAR=Neg
#Charmless#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#Charmless#	echo cp -av ${template} ${thisToml}
#Charmless#	cp -av ${template} ${thisToml}
#Charmless#     find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;    
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#Charmless#	# Run the code to generate the full file
#Charmless#	echo ${pythonExe} ${writeScript} ${thisToml}
#Charmless#	${pythonExe} ${writeScript} ${thisToml}
#Charmless#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#Charmless#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#Charmless#	cd ${dataDir}
#Charmless#	# and run the code to generate the plots
#Charmless#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	cd -
#Charmless#    done
#Charmless#    # Xi(dss)
#Charmless#    OP=doublet.2fl
#Charmless#    UN=doublet_2fl
#Charmless#    qqq=ssu
#Charmless#    had='\Xi(dss)'
#Charmless#    for RAT in G;
#Charmless#    do
#Charmless#	PAR=Pos
#Charmless#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#Charmless#	echo cp -av ${template} ${thisToml}
#Charmless#	cp -av ${template} ${thisToml}
#Charmless#     find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#Charmless#	# Run the code to generate the full file
#Charmless#	echo ${pythonExe} ${writeScript} ${thisToml}
#Charmless#	${pythonExe} ${writeScript} ${thisToml}
#Charmless#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#Charmless#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#Charmless#	cd ${dataDir}
#Charmless#	# and run the code to generate the plots
#Charmless#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	cd -
#Charmless#	# Negative parity
#Charmless#	PAR=Neg
#Charmless#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#Charmless#	echo cp -av ${template} ${thisToml}
#Charmless#	cp -av ${template} ${thisToml}
#Charmless#     find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;    
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#Charmless#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#Charmless#	# Run the code to generate the full file
#Charmless#	echo ${pythonExe} ${writeScript} ${thisToml}
#Charmless#	${pythonExe} ${writeScript} ${thisToml}
#Charmless#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#Charmless#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#Charmless#	cd ${dataDir}
#Charmless#	# and run the code to generate the plots
#Charmless#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#Charmless#	cd -
#Charmless#    done
    # Sigma_c(udc)
    OP=sigma12.3fl
    UN=sigma12_3fl
    qqq=udc
    had='\Sigma_{c}(udc)'
    for RAT in G;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	echo ${finalToml}
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # xi^\prime(usc)
    OP=sigma12.3fl
    UN=sigma12_3fl
    qqq=usc
    had='\Xi_{c}^{\prime}(usc)'
    for RAT in G;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # Omega_{c}(ssc)
    OP=doublet.2fl
    UN=doublet_2fl
    qqq=ssc
    had='\Omega_{c}(ssc)'
    for RAT in G;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # Sigma(uus)
    OP=lambda.3fl
    UN=lambda_3fl
    qqq=udc
    had='\Lambda_{c}(udc)'
    for RAT in G;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # xi_{c}(usc
    OP=lambda.3fl
    UN=lambda_3fl
    qqq=usc
    had='\Xi_{c}(usc)'
    for RAT in G;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # xi_{cc}(ccu)
    OP=doublet.2fl
    UN=doublet_2fl
    qqq=ccu
    had='\Xi_{cc}(ccu)'
    for RAT in G;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
    # \Omega_cc (ccs)
    OP=doublet.2fl
    UN=doublet_2fl
    qqq=ccs
    had='\Omega_{cc}(ccs)'
    for RAT in G;
    do
	PAR=Pos
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
	# Negative parity
	PAR=Neg
	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
	echo cp -av ${template} ${thisToml}
	cp -av ${template} ${thisToml}
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
	# Run the code to generate the full file
	echo ${pythonExe} ${writeScript} ${thisToml}
	${pythonExe} ${writeScript} ${thisToml}
	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
	cd ${dataDir}
	# and run the code to generate the plots
	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
	${pythonExe} ${scriptDir}/${exe} ${finalToml}
	cd -
    done
#J32#    #########################3
#J32#    # 3/2
#J32#    ##########################
#J32#    # 3/2
#J32#    # Sigma_{c} udc
#J32#    OP=sigma32.3fl
#J32#    UN=sigma32_3fl
#J32#    qqq=udc
#J32#    had='\Sigma_{c}(udc)'
#J32#    for RAT in G;
#J32#    do
#J32#	PAR=Pos
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#	# Negative parity
#J32#	PAR=Neg
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#    done
#J32#    # xi_{c}(usc)
#J32#    OP=sigma32.3fl
#J32#    UN=sigma32_3fl
#J32#    qqq=usc
#J32#    had='\Xi_{c}(usc))'
#J32#    for RAT in G;
#J32#    do
#J32#	PAR=Pos
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#	# Negative parity
#J32#	PAR=Neg
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#    done
#J32#    # quadruplet omega_c ssc
#J32#    OP=quadruplet.2fl
#J32#    UN=quadruplet_2fl
#J32#    qqq=ssc
#J32#    had='\Omega_{c}(ssc)'
#J32#    for RAT in G;
#J32#    do
#J32#	PAR=Pos
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#	# Negative parity
#J32#	PAR=Neg
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#    done
#J32#    # xi_{cc} (ccu)
#J32#    OP=quadruplet.2fl
#J32#    UN=quadruplet_2fl
#J32#    qqq=ccu
#J32#    had='\Xi_{cc}(ccu)'
#J32#    for RAT in G;
#J32#    do
#J32#	PAR=Pos
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#	# Negative parity
#J32#	PAR=Neg
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#    done
#J32#    # omega_{cc} (ccs)
#J32#    OP=quadruplet.2fl
#J32#    UN=quadruplet_2fl
#J32#    qqq=ccs
#J32#    had='\Omega_{cc}(ccs)'
#J32#    for RAT in G;
#J32#    do
#J32#	PAR=Pos
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#	# Negative parity
#J32#	PAR=Neg
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#    done
#J32#    #omega_ccc
#J32#    OP=quadruplet.1fl
#J32#    UN=quadruplet_1fl
#J32#    qqq=ccc
#J32#    had='\Omega_{ccc}(ccc)'
#J32#    for RAT in G;
#J32#    do
#J32#	PAR=Pos
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#	# Negative parity
#J32#	PAR=Neg
#J32#	thisToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml
#J32#	echo cp -av ${template} ${thisToml}
#J32#	cp -av ${template} ${thisToml}
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~EXT~ReconCorr~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~OP~${OP}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~UN~${UN}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~QQQ~${qqq}~" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~NAME~${had}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~RAT~${RAT}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~${RAT} = false~${RAT} = true~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~PAR~${PAR}~g" {} \;
#J32#	find ${outTomlDir} -type f -name ${UN}_${qqq}_${temp}_${RAT}_${PAR}-base.toml -exec sed -i.bak "s~negParity = false~negParity = true~g" {} \;
#J32#	# Run the code to generate the full file
#J32#	echo ${pythonExe} ${writeScript} ${thisToml}
#J32#	${pythonExe} ${writeScript} ${thisToml}
#J32#	# This puts the toml file in AutoToml-base/OP_QQQ_cool_RAT_PAR-base.toml
#J32#	finalToml=${outTomlDir}/${UN}_${qqq}_${temp}_${RAT}_${PAR}_Ratios.toml
#J32#	cd ${dataDir}
#J32#	# and run the code to generate the plots
#J32#	echo ${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	${pythonExe} ${scriptDir}/${exe} ${finalToml}
#J32#	cd -
#J32#    done
#J32#    
done
