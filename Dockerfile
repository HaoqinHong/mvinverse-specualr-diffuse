ARG BASE_IMAGE=da3-base-3dat
FROM ${BASE_IMAGE}

ENV TRITON_CACHE_DIR=/tmp/triton_cache \
    PYTHONUNBUFFERED=1

RUN mkdir -p /tmp/triton_cache

WORKDIR /workspace/code/mvinverse
COPY . /workspace/code/mvinverse

# Set proxies for apt/pip and most CLI tools (both uppercase and lowercase are used by different tools).
ARG HTTP_PROXY=http://proxy.ubisoft.org:3128
ARG HTTPS_PROXY=http://proxy.ubisoft.org:3128
ARG NO_PROXY=127.0.0.0/8,localhost,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,169.254.0.0/16,.ubisoft.fr,.ubisoft.org,.ubisoft.onbe

ENV HTTP_PROXY=${HTTP_PROXY} \
    HTTPS_PROXY=${HTTPS_PROXY} \
    NO_PROXY=${NO_PROXY} \
    http_proxy=${HTTP_PROXY} \
    https_proxy=${HTTPS_PROXY} \
    no_proxy=${NO_PROXY}

RUN printf 'Acquire::http::Proxy "%s";\nAcquire::https::Proxy "%s";\n' "$HTTP_PROXY" "$HTTPS_PROXY" > /etc/apt/apt.conf.d/99proxy

# Install project dependencies at build time. When code is bind-mounted, entrypoint can refresh installs.
RUN python3 -m pip install -U pip setuptools wheel \
    && python3 -m pip install -r requirements.txt \
    && python3 -m pip install -e .

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV CODE_PATH=/workspace/code/mvinverse \
    TRAINING_DIR=training \
    CONFIG_NAME= \
    NPROC_PER_NODE=1 \
    INSTALL_DEPS_ON_START=1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["train"]