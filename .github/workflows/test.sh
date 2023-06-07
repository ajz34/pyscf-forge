#!/usr/bin/env bash
  
set -e

cd ./pyscf

# ajz34: RS_PBE and rsdh is added from dh
pytest -k 'not _slow and not RS_PBE and not rsdh'
