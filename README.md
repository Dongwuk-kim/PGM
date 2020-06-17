- numba class 구현을 위해 따로 판 브랜치입니다.

### File 설명
- BPM_MF_algo_numba_class.py
    - 기존 코드에서 typo좀 수정했고, spec을 타입에 맞추어 수정함.
- test_algo_numba_class.ipynb
    - 여기 노트북 파일에서 따로 test 돌리는 중이었음.

### Comment
- @jitclass 선언을 안 하고 쓰면 에러 안나고 잘 돌아가는 것까지 확인함.
- @jitclass 선언을 하면 에러가 나는데, 이것저것 시도해보다가 다음의 문제일 것으로 보임.
    - numba는 sparse_coo_matrix를 지원을 안 함. 따라서 problem.data_m 과 problem.test_m이 sparse_coo_matrix인게 문제를 일으키지 않나 싶음. 그냥 바로 row, col index와 data값을 넣어줘야할 것 같음.
- spec과 `__init__` 에서 선언한 값들의 dtype이 정확히 일치해야함. int, float 구분 다 중요한 거 같음. 예를 들어서 float로 선언 되어있는데 R=5를 대입하면 오류남. R=5.0 이런 식으로 대입해야함. 예상되는 문제 중에 이러한 변수 type 선언에 의한게 있지 않을까 싶음.