ln -s "${DESTINATION}"/licenses ~/Licenses
. "${COMPILER}/bin/compilervars.sh" intel64
export LD_LIBRARY_PATH="${PWD}/.travis:${COMPILER}/ism/bin/intel64:${COMPILER}/lib/intel64_lin:${LD_LIBRARY_PATH}"
