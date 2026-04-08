FROM nvcr.io/nvidia/isaac-lab:2.3.2

# Make python/python3 available everywhere
RUN printf '#!/bin/bash\nexec /isaac-sim/python.sh "$@"\n' > /usr/local/bin/python && \
    chmod +x /usr/local/bin/python && \
    ln -sf /usr/local/bin/python /usr/local/bin/python3

