#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
    version="2021.4.689"
else
    version="$1"
fi

scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Retrieve OpenVINO checksum.
curl -sSL https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021 > $scriptdir/GPG-PUB-KEY-INTEL-OPENVINO-2021
echo "5f5cff8a2d26ba7de91942bd0540fa4d $scriptdir/GPG-PUB-KEY-INTEL-OPENVINO-2021" > $scriptdir/CHECKSUM
md5sum --check $scriptdir/CHECKSUM

# Add OpenVINO repository (deb).
sudo apt-key add $scriptdir/GPG-PUB-KEY-INTEL-OPENVINO-2021
echo "deb https://apt.repos.intel.com/openvino/2021 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2021.list
sudo apt update

# Install OpenVINO package.
sudo apt install -y intel-openvino-runtime-ubuntu20-$version
