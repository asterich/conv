

### 32768, 3, block 16

- naive: 13.766
- omp: 0.268222 0.254773
- mkl_dnn: 0.202492 0.197315
- omp_blocking: 0.249234 0.23751

- naive: 13.766
- omp: 0.290167 0.292624 0.298105
- mkl_dnn: 0.197232 0.197802 0.198295
- omp_blocking_simd: 0.25997 0.219818 0.218009

### 16384, 3, block 16

- naive: 3.43915 3.44039
- omp: 0.0951515 0.108397
- mkl_dnn: 0.0503746 0.0498817
- omp_blocking: 0.0592411 0.059491

- naive: 3.443 3.43911 3.44328
- omp: 0.125233 0.12594 0.121418
- mkl_dnn: 0.050496 0.0502221 0.0506664
- omp_blocking_simd: 0.052553 0.0524888 0.0525265

### 16384, 9, block 16

block 4:
- naive: 7.8934 7.89349 7.89721 7.90109
- omp: 0.259619 0.200907 0.197118 0.182417
- mkl_dnn: 0.157691 0.157588 0.15803 0.157878
- omp_blocking: 0.152608 0.132943 0.132917 0.132984

block 8:
- naive: 7.90015 7.89149 7.89016 7.8946
- omp: 0.208322 0.199848 0.200874 0.197944
- mkl_dnn: 0.157806 0.157588 0.15758 0.157479
- omp_blocking: 0.12881 0.128304 0.128574 0.129004

block 16:
- naive: 7.89424 7.88535 7.89162
- omp: 0.193943 0.19838 0.186076
- mkl_dnn: 0.158396 0.157648 0.157495
- omp_blocking: 0.133212 0.141089 0.133016

block 32:
- naive: 7.89109 7.89011 7.89278
- omp: 0.198153 0.195112 0.233713
- mkl_dnn: 0.157967 0.157595 0.15857
- omp_blocking: 0.168917 0.167422 0.167718

block 64:
- naive: 7.90263 7.89665
- omp: 0.2558 0.193164
- mkl_dnn: 0.157609 0.157593
- omp_blocking: 0.162419 0.164397

### 32768, 9, block 16

- naive: 31.6417 31.6522
- omp: 0.60554 0.596949
- mkl_dnn: 0.654082 0.653197
- omp_blocking: 0.536109 0.555612