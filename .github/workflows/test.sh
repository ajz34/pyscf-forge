#!/usr/bin/env bash
  
set -e

cd ./pyscf

# ajz34: RS_PBE and rsdh is added from dh
pytest -k 'not _slow and not _RS_ and not rsdh and not _D3 and not _GAUSSIAN'
