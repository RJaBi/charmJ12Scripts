#!/bin/bash

pythonExe=python
dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
scriptDir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../code/bin/
exe=simpleFit.py

tomlSave=AutoToml_fits_03_03_23

echo "dir IS ${dir}"
echo "scriptDir IS ${scriptDir}"

tomlDir=${dir}/${tomlSave}/
mkdir -v ${tomlDir}

for NT in 128 32 36 40 48 56 64; do
#for NT in 128; do
    #for NT in 64; do

    OP=doublet_2fl
    # Do the CCU separately as it has our plot in it
    # Copy the example to the folder
    particle=ccu
    thisToml=${tomlDir}/${NT}_${OP}_${particle}.toml
    echo cp -av ${dir}/fitTemplates/${NT}_${OP}_ccu.toml ${thisToml}
    cp -av ${dir}/fitTemplates/${NT}_${OP}_ccu.toml ${thisToml}
    # and now modify it with the quark content
    find ${tomlDir} -type f -name ${NT}_${OP}_${particle}.toml -exec sed -i "s~OP~${particle}~g" {} \;
    # and run the code
    echo ${pythonExe} ${scriptDir}/${exe} ${thisToml}
    ${pythonExe} ${scriptDir}/${exe} ${thisToml}
    
    # The 3 flavour operators
    for OP in sigma12_3fl lambda_3fl; do # sigma32_3fl
	# Make a folder to put the modified toml in
	tomlExample=${dir}/fitTemplates/${NT}_${OP}_OP.toml
	#for particle in uud uus uuc ssu ssc ccu ccs
	#for particle in uds udc usc; do
	for particle in udc usc; do
	#for particle in ccc
	    # Copy the example to the folder
	    thisToml=${tomlDir}/${NT}_${OP}_${particle}.toml
	    echo cp -av ${tomlExample} ${thisToml}
	    cp -av ${tomlExample} ${thisToml}
	    # and now modify it with the quark content
	    find ${tomlDir} -type f -name ${NT}_${OP}_${particle}.toml -exec sed -i "s~OP~${particle}~g" {} \;
	    # and run the code
	    echo ${pythonExe} ${scriptDir}/${exe} ${thisToml}
	    ${pythonExe} ${scriptDir}/${exe} ${thisToml}
	done
    done
    # The two flavour operators
    for OP in doublet_2fl; do # quadruplet_2fl
	# Make a folder to put the modified toml in
	tomlExample=${dir}/fitTemplates/${NT}_${OP}_OP.toml
	#tomlDir=${dir}/${tomlSave}/
	#mkdir -v ${tomlDir}      
	#for particle in uud uus uuc ssu ssc ccu ccs; do
	for particle in uuc ssc ccs; do
	    # Copy the example to the folder
	    thisToml=${tomlDir}/${NT}_${OP}_${particle}.toml
	    echo cp -av ${tomlExample} ${thisToml}
	    cp -av ${tomlExample} ${thisToml}
	    # and now modify it with the quark content
	    find ${tomlDir} -type f -name ${NT}_${OP}_${particle}.toml -exec sed -i "s~OP~${particle}~g" {} \;
	    # and run the code
	    echo ${pythonExe} ${scriptDir}/${exe} ${thisToml}
	    ${pythonExe} ${scriptDir}/${exe} ${thisToml}
	done
    done

    # The 1 flavour operators
    for OP in ; do # quadruplet_1fl
	# Make a folder to put the modified toml in
	tomlExample=${dir}/fitTemplates/${NT}_${OP}_OP.toml
	#tomlDir=${dir}/${tomlSave}/
	#mkdir -v ${tomlDir}      
	for particle in ccc; do
	    # Copy the example to the folder
	    thisToml=${tomlDir}/${NT}_${OP}_${particle}.toml
	    echo cp -av ${tomlExample} ${thisToml}
	    cp -av ${tomlExample} ${thisToml}
	    # and now modify it with the quark content
	    find ${tomlDir} -type f -name ${NT}_${OP}_${particle}.toml -exec sed -i "s~OP~${particle}~g" {} \;
	    # and run the code
	    echo ${pythonExe} ${scriptDir}/${exe} ${thisToml}
	    ${pythonExe} ${scriptDir}/${exe} ${thisToml}
	done
    done
    
done
