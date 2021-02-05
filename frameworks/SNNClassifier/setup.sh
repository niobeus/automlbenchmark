#!/usr/bin/env bash
HERE=$(dirname "$0")
AMLB_DIR="$1"
VERSION=${2:-"stable"}
REPO=${3:-"https://github.com/scikit-learn/scikit-learn.git"}
PKG=${4:-"scikit-learn"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

# by passing the module directory to `setup.sh`, it tells it to automatically create a virtual env under the current module.
# this virtual env is then used to run the exec.py only, and can be configured here using `PIP` and `PY` commands.
. ${HERE}/../shared/setup.sh ${HERE}
#. ${AMLB_DIR}/frameworks/shared/setup.sh ${HERE}

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
    PIP install --no-cache-dir -U torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    PIP install  --no-cache-dir -U numpy
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
    PIP install --no-cache-dir -U torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    PIP install  --no-cache-dir -U numpy
else
#    PIP install --no-cache-dir -e git+${REPO}@${VERSION}#egg=${PKG}
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}
    PIP install --no-cache-dir -U torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    PIP install  --no-cache-dir -U numpy
fi
