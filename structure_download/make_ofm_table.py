import time
import pandas as pd
import os
import json
from pymatgen.core.structure import Structure
from matminer.featurizers.structure import OrbitalFieldMatrix
import warnings

# spglib에서 오는 DeprecationWarning만 억제
warnings.filterwarnings('ignore', category=DeprecationWarning, module='spglib.*')
start = time.perf_counter()




# 입력 디렉토리 설정
#prop = "exp_band_gap"
prop = "exp_formation_enthalpy"
#prop = "hse06_band_gap"
#prop = "pbe_+u_band_gap"
input_directory = f"./{prop}/structure_cif"
output_directory = f"./{prop}/{prop}.csv"
ids_directory = "../../ids_target_updated.json"





# 디렉토리 존재 여부 확인
if not os.path.exists(input_directory):
    print(f"디렉토리 '{input_directory}'가 존재하지 않습니다.")

    exit()
else:
    print(f"디렉토리 '{input_directory}'가 존재합니다.")

if not os.path.exists(ids_directory):
    print(f"파일 '{ids_directory}'가 존재하지 않습니다.")
    exit()
else:
    print(f"파일 '{ids_directory}'가 존재합니다.")


with open(ids_directory, "r", encoding="utf-8-sig") as f:
    ids = json.load(f)

prop_dict = ids[prop]
filename_in_ref = prop_dict["file_name"]
# filename_in_ref_set = set(filename_in_ref)
target_in_ref = prop_dict["target"]


# .cif 파일 읽기 및 구조 리스트 생성
structure = []
id = []
target = []
for filename in os.listdir(input_directory):
    filename_matched = False
    if filename.endswith(".cif"):
        filepath = os.path.join(input_directory, filename)
        structure.append(Structure.from_file(filepath))
        id.append(filename)
        target_value = target_in_ref[filename_in_ref.index(filename)]
        target.append(target_value)
        filename_matched = True
    
    if not filename_matched:
        print(f"파일 '{filename}'은 '{filename_in_ref}'에 존재하지 않습니다.")
        exit()




# 데이터프레임 생성
df = pd.DataFrame({'structure': structure, 'id': id, 'target' : target})


setting = time.perf_counter()
print('테이블 세팅 소요시간 = {:.2f} sec'.format(setting-start))

# 동일한 DataFrame 사용
# df['structure'] = df['structure'].apply(lambda x: x.to_json())


# OFM 변환 전 테스트로 데이터 저장
# df.to_csv('test.csv', index=False)
# table = pa.Table.from_pandas(df, preserve_index=False)
# pq.write_table(table, 'data.parquet', compression='snappy')

# OrbitalFieldMatrix 적용
ofm = OrbitalFieldMatrix(period_tag=False)
df = ofm.featurize_dataframe(df, 'structure', ignore_errors=True)
df.drop(columns=['structure'], inplace=True)

convert = time.perf_counter()
print('데이터 변환 소요시간 = {:.2f} sec'.format(convert-setting))


# 결과를 CSV로 저장
# output_file = 'structure.csv'
df.to_csv(output_directory, index=False)
print(f"'{output_directory}' 파일이 생성되었습니다.")


end = time.perf_counter()

print('파일 저장 소요시간 = {:.2f} sec'.format(end-convert))
print('전체 소요시간 = {:.2f} sec'.format(end-start))
