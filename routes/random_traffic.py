#!/usr/bin/env python3
"""
generate_random_routes_connected.py

네트워크(.net.xml)에 대해 ‘실제로 주행 가능한’ 무작위 trip들을 만든 뒤
<trip …/> 목록이 들어 있는 .rou.xml 을 출력한다.

사용 예:
    python generate_random_routes_connected.py \
        -n sejong_area6.net.xml \
        -o random_traffic.rou.xml \
        -b 0       \
        -e 3600    \
        -t 1000    \
        --seed 42
"""

import argparse
import random
import sys
import xml.etree.ElementTree as ET

try:
    import sumolib  # SUMO가 설치돼 있으면 기본 제공
except ImportError:
    sys.exit("sumolib 를 찾을 수 없습니다.  "
             "SUMO_HOME 환경변수를 설정했는지 확인하세요.")

###############################################################################
# 인자 파서
###############################################################################
def parse_args():
    p = argparse.ArgumentParser(description="Generate random *connected* trips")
    p.add_argument('-n', '--net-file',   required=True, help='.net.xml path')
    p.add_argument('-o', '--output',     required=True, help='.rou.xml out path')
    p.add_argument('-b', '--begin',      type=float, default=0.0,    help='earliest depart (s)')
    p.add_argument('-e', '--end',        type=float, default=3600.0, help='latest depart (s)')
    p.add_argument('-t', '--num-trips',  type=int,   default=1000,   help='trip count')
    p.add_argument('--seed',             type=int,   default=None,   help='random seed')
    p.add_argument('--attempt-factor',   type=int,   default=5,
                   help='max attempts per wanted trip (default 5×)')
    return p.parse_args()

###############################################################################
# 유틸
###############################################################################
###############################################################################
# lane-helper ― 어떤 버전이든 ‘승용차(passenger) 통행 가능?’ 판단
###############################################################################
def lane_allows_passenger(lane) -> bool:
    # 1) allowed 리스트 얻기
    if hasattr(lane, "getAllowed"):
        allowed = lane.getAllowed()              # 신버전
    else:
        allowed = getattr(lane, "_allowed", [])  # 구버전

    # 2) disallowed 리스트 얻기 (allowed 가 비어 있을 때 대비)
    if hasattr(lane, "getDisallowed"):
        disallowed = lane.getDisallowed()
    else:
        disallowed = getattr(lane, "_disallowed", [])

    # 규칙:
    #  - allowed 리스트가 ‘비어 있으면’ 전체 허용 (SUMO 기본)
    #  - 그 외엔 passenger 가 명시적으로 허용돼야 한다
    if not allowed:
        return 'passenger' not in disallowed
    return 'passenger' in allowed or 'all' in allowed

def is_driveable(edge):
    """Driveable edge = internal 이 아니고, 최소 1개 lane 이 승용차 허용"""
    if edge.getFunction() == 'internal':
        return False
    return any(lane_allows_passenger(ln) for ln in edge.getLanes())

###############################################################################
# 메인
###############################################################################
def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    # ---------- 1. 네트워크 로드
    try:
        net = sumolib.net.readNet(args.net_file)
    except Exception as e:
        sys.exit(f"네트워크 파일 읽기 실패: {e}")

    # ---------- 2. 출발/도착 후보 edge 수집
    cand_edges = [e for e in net.getEdges() if is_driveable(e)]
    if len(cand_edges) < 2:
        sys.exit("driveable edge 가 2개 미만입니다.")

    # ---------- 3. trip 생성
    trips = []
    max_attempts = args.attempt_factor * args.num_trips
    attempts = 0

    
    while len(trips) < args.num_trips and attempts < max_attempts:
        attempts += 1
        orig, dest = random.sample(cand_edges, 2)
    
        # 경로 존재 여부 확인
        try:
            path, _ = net.getShortestPath(orig, dest, vClass='passenger', weight='time')
        except TypeError:
                # 아주 구버전(sumolib<1.4)은 서명( weight, maxCost, vClass )이므로 fallback
                path, _ = net.getShortestPath(orig, dest, vClass='passenger')
                
        if not path:        # 연결 안 되어 있으면 skip
            continue

        depart = random.uniform(args.begin, args.end)
        trips.append((depart, orig.getID(), dest.getID()))

    if len(trips) < args.num_trips:
        print(f"⚠️   {args.num_trips}개 중 {len(trips)}개만 생성했습니다 "
              f"(연결 가능한 pair 부족).", file=sys.stderr)

    # ---------- 4. .rou.xml 작성
    trips.sort(key=lambda x: x[0])  # 시간순으로 정렬(선택 사항)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<routes>\n')
        f.write('  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" '
                'length="5.0" maxSpeed="70"/>\n\n')

        for i, (depart, frm, to) in enumerate(trips):
            f.write(f'  <trip id="trip_{i}" type="car" depart="{depart:.2f}" '
                    f'from="{frm}" to="{to}" departLane="best"/>\n')

        f.write('</routes>\n')

    print(f"✅  {len(trips)} trips written to '{args.output}' "
          f"(attempts: {attempts})")

if __name__ == "__main__":
    main()
