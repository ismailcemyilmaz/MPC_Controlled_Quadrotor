#!/bin/bash
# Persistent dependency setup for MPC project
# Installs into /shared-workspace/ so it survives Docker restarts
#
# First time:  bash setup_deps.sh install
# After restart: bash setup_deps.sh   (or source env_setup.sh)

ACADOS_DIR="/shared-workspace/acados"
PIP_TARGET="/shared-workspace/pip-packages"

install_deps() {
    echo "=== Installing CasADi + acados to shared workspace ==="

    # CasADi
    pip install --target="$PIP_TARGET" casadi numpy
    echo "[OK] casadi installed to $PIP_TARGET"

    # acados
    if [ ! -d "$ACADOS_DIR" ]; then
        echo "Cloning acados..."
        git clone https://github.com/acados/acados.git "$ACADOS_DIR"
        cd "$ACADOS_DIR" && git submodule update --recursive --init
    fi

    cd "$ACADOS_DIR"
    mkdir -p build && cd build
    cmake .. -DACADOS_WITH_QPOASES=ON
    make -j4
    sudo make install
    pip install --target="$PIP_TARGET" "$ACADOS_DIR/interfaces/acados_template"

    # Tera renderer (needed for C code generation)
    TERA_BIN="$ACADOS_DIR/bin/t_renderer"
    if [ ! -f "$TERA_BIN" ]; then
        mkdir -p "$ACADOS_DIR/bin"
        TERA_URL="https://github.com/acados/tera_renderer/releases/download/v0.2.0/t_renderer-v0.2.0-linux-amd64"
        echo "Downloading tera renderer..."
        wget -q -O "$TERA_BIN" "$TERA_URL" && chmod +x "$TERA_BIN"
        echo "[OK] tera renderer installed"
    fi

    echo "[OK] acados built and installed"
}

setup_env() {
    export PYTHONPATH="$PIP_TARGET:$PYTHONPATH"
    export ACADOS_SOURCE_DIR="$ACADOS_DIR"
    export LD_LIBRARY_PATH="$ACADOS_DIR/lib:$LD_LIBRARY_PATH"
    echo "[env] PYTHONPATH, ACADOS_SOURCE_DIR, LD_LIBRARY_PATH set"
}

# Generate a small source-able env file
cat > /shared-workspace/src/mpc-quadrotor/env_setup.sh << 'ENVEOF'
export PYTHONPATH="/shared-workspace/acados/interfaces/acados_template:/shared-workspace/pip-packages:$PYTHONPATH"
export ACADOS_SOURCE_DIR="/shared-workspace/acados"
export LD_LIBRARY_PATH="/shared-workspace/acados/lib:$LD_LIBRARY_PATH"
ENVEOF

if [ "$1" = "install" ]; then
    install_deps
fi

setup_env
echo ""
echo "Ready. Before using this project, run:"
echo "  source /shared-workspace/src/mpc-quadrotor/env_setup.sh"
