// Korean interface (PLAN §16, D4). Keys are the English strings, so English needs no table and stays exactly as before.
// Language: &lang=ko / &lang=en in the address, else the browser's language; the header button switches it.

const KO = {
  // static page (index.html, data-i18n)
  'Loading…': '불러오는 중…',
  'Controls': '설정',
  'Showcase': '대표 장면',
  'All test photos': '전체 테스트 사진',
  'Sort': '정렬',
  'by photo id': '사진 번호순',
  'most houses cut off': '고립 가구 많은 순',
  'most disagreements': '불일치 많은 순',
  'lowest mean IoU': '평균 IoU 낮은 순',
  'Copy link to this view': '이 화면 링크 복사',
  // your own photo (PLAN §16.10): the small model runs in this browser
  'Your photo': '내 사진', 'Try your own drone photo': '내 드론 사진으로 해 보기',
  'A small model runs in this browser: the photo never leaves your computer (first use downloads 27 MB).':
    '작은 모델이 이 브라우저에서 실행됩니다: 사진은 컴퓨터 밖으로 나가지 않습니다 (처음 사용할 때 27 MB를 내려받습니다).',
  'portrait photo rotated to landscape': '세로 사진을 가로로 회전', 'cropped to the centre': '가운데를 잘라냄',
  'small photo upscaled: expect a blurrier map': '작은 사진을 확대함: 지도가 더 흐릴 수 있음',
  'Preparing the photo…': '사진 준비 중…', 'Loading the model (first use: 27 MB)…': '모델 불러오는 중 (처음: 27 MB)…',
  'Mapping the photo in this browser…': '이 브라우저에서 사진 지도 만드는 중…',
  'The photo could not be mapped': '사진 지도를 만들지 못했습니다',
  'Map': '지도',
  'Prediction': '예측',
  'Labelled map': '라벨 지도',
  'Swipe': '비교',
  'Classes': '클래스',
  'Water probability': '물 확률',
  'Overlay': '오버레이',
  'House status': '가구 상태',
  'Disagreements with the labelled map': '라벨 지도와의 불일치',
  'Entry points': '진입 지점',
  'Risk-aware truck (avoid likely water)': '위험 회피 트럭 (물 가능성 지역 회피)',
  'Colour-blind palette': '색각 이상 팔레트',
  'Houses': '가구',
  'Download house list (CSV)': '가구 목록 내려받기 (CSV)',
  'Entry point': '진입 지점',
  'Rescue team enters:': '구조대 진입:',
  'Choose on map': '지도에서 선택',
  'Auto': '자동',
  'Legend': '범례',
  'Reachable by truck (confident)': '트럭으로 도달 가능 (확실)',
  'Probably by truck: check first': '트럭 가능성 있음: 먼저 확인',
  'Boat needed': '보트 필요',
  'Last stretch on foot': '마지막 구간 도보',
  'No route: air team': '경로 없음: 항공 구조',
  'Truck': '트럭', 'Boat': '보트', 'Foot': '도보',
  'Route on the other map, when they disagree': '두 지도가 다를 때 다른 지도의 경로',
  'Entry point (truck or boat)': '진입 지점 (트럭 또는 보트)',
  'About': '소개',
  'Previous photo': '이전 사진', 'Next photo': '다음 사진', 'Dataset': '데이터셋', 'Sort the test photos': '테스트 사진 정렬',
  'Test photos': '테스트 사진', 'Map source': '지도 종류', 'Overlay opacity': '오버레이 불투명도',
  'Houses in this photo: arrow keys to move, Enter to send a team': '이 사진의 가구: 화살표 키로 이동, Enter로 구조대 보내기',
  'Drone photo with flood map': '침수 지도가 표시된 드론 사진',
  'Drag to compare the prediction and the labelled map': '끌어서 예측과 라벨 지도 비교',
  'The same router on the human labels': '사람이 만든 라벨에 같은 경로 탐색기를 적용한 결과',
  "The model's water probability per cell (prediction only)": '셀별 모델의 물 확률 (예측만)',
  'Every house of this photo with its status, tier, minutes and metres per mode (prediction; labelled map alongside)':
    '이 사진의 모든 가구: 상태, 신뢰 등급, 소요 시간(분), 수단별 거리(m) (예측; 라벨 지도 함께)',
  'About text': `드론 사진의 각 픽셀을 같은 라벨 사진으로 학습한 두 모델 중 하나가 다섯 가지 클래스로 분류합니다: 이
    프로젝트가 출발한 퓨샷 모델 <b>Chameleon</b>과 일반적인 분할 모델(뒤집은 영상을 평균하는 DeepLabV3-ResNet50 3개
    앙상블)입니다. 학습에는 라벨이 있는 사진 <b>50장</b>(FloodNet) 또는 20장(RescueNet)을 사용했고,
    데이터셋마다 라벨이 있는 검증 사진 30장으로 임계값과 경로 설정을 골랐습니다. 각 데이터셋은 그 데이터에서 더 나은
    모델로 열리고(아래 참조), '모델' 전환으로 다른 모델을 볼 수 있습니다. 이어서 4×4 픽셀 셀 위에서 경로 탐색기가 진입 지점에서 각 가구까지 가장 싼 길을
    찾습니다: 마른 땅은 트럭(도로, 또는 비용 4배의 마당; 물일 가능성이 높은 곳은 제외), 물은 보트, 마지막 구간은 도보
    (다시 차량으로 돌아가지 않음). 차량에는 폭이 있어 트럭은 건물 사이 약 2.5 m, 보트는 약 1.6 m가 필요하므로 두 집
    사이 틈은 지나가지 못하고 사람은 걸어서 지나갈 수 있습니다. 라벨 지도는 사람이 만든 라벨에 같은 경로 탐색기를 적용한
    것입니다: 데이터셋은 픽셀에 라벨을 달았을 뿐, 도달 가능성에 라벨을 단 것이 아닙니다.`,
  'About confidence': `신뢰도: 예측 지도를 두 번 경로 탐색합니다. 보통 규칙(FloodNet은 p_wet &lt; 0.3, RescueNet은 0.2인
    땅만 도로 밖 주행 허용)과 신중한 규칙(p_wet &lt; 0.05 / 0.15)입니다. 신중한 지도에서도 트럭이 닿는 가구는 '확실'
    (FloodNet 테스트 사진에서 앙상블 99 %, Chameleon 100 % 실제로 트럭 도달 가능)이고, 나머지는 '먼저 확인'(82 % /
    79 %)으로 표시합니다. FloodNet 전체 테스트에서 앙상블이 트럭 도달 가능으로 표시한 가구의 약 7 %(Chameleon 11 %)는
    실제로는 그렇지 않았습니다(대부분 모델이 놓친 물을 지나는 경로). 소요 시간과
    거리는 대략적인 추정입니다(도로 트럭 20 km/h, 도로 밖 5, 보트 6, 도보 3; 축척은 사진 속 자동차 기준).`,
  'About numbers': `FloodNet 전체 448장 테스트: DeepLabV3 앙상블 mIoU .62, Chameleon .51 (라벨 1,445장으로 학습한 PSPNet:
    같은 클래스에서 .85). 같은 사진으로 미리 등록한 비교(Phase 2)에서 Chameleon은 라벨이 아주 적을 때(FloodNet 사진
    1-5장)와, 조정용 사진 30장이 도움이 되는 새 재난의 사진 10-20장에서만 앞섰습니다: 여기의 RescueNet(사진 20장)에서
    mIoU .42, DeepLabV3 앙상블은 .37(평가용 100장)이므로 RescueNet은 Chameleon으로 열립니다. 두 모델 모두 같은 경로
    탐색기와 임계값을 쓰고, 각자의 상태, 신뢰 등급, 사진별 IoU를 보여 줍니다. 다른 모델이 다르게 판단한 가구는 가구
    목록에 표시되고 지도 아래에 개수가 나오며, '두 모델이 가장 다른 순' 정렬로 그런 사진을 먼저 볼 수 있습니다. 대표 장면
    하나는 Chameleon의 정직한 실패 사례입니다: 얕은 물을 마른 땅으로 읽었습니다.`,
  'About upload': `'내 드론 사진으로 해 보기'는 작은 모델(LR-ASPP-MobileNetV3, 13 MB, 같은 FloodNet 라벨 사진 50장과
    기준 모델이 스스로 라벨을 단 라벨 없는 사진으로 학습: 전체 테스트 mIoU .60)로 이 브라우저에서 사진의 지도를 만들고 같은 규칙으로 경로를 찾습니다. 사진은 어디에도
    보내지지 않습니다. 라벨이 없으므로 가구와 진입 지점은 모델의 지도에서 나오며, FloodNet과 비슷한 높이에서 바로 아래로
    찍은 드론 사진에 맞습니다.`,
  'Source code on GitHub': 'GitHub의 소스 코드',
  'About source': `소스 코드, 평가 데이터 분할, 모든 수치의 측정 방법:
    <a href="https://github.com/MarlenMM/rescue-router" target="_blank" rel="noopener">github.com/MarlenMM/rescue-router</a>.
    연구용 시연이며 실제 구조 작전의 결정 도구가 아닙니다.`,
  'About browse': `8개 대표 장면 외에 평가 세트의 테스트 사진 200장(FloodNet 100, RescueNet 100) 전체를 볼 수 있으며,
    우리가 고른 것이 아닙니다. 주황색 숫자는 예측 지도에서 트럭이 닿지 못하는 가구 수, 분홍색 점은 예측과 라벨 지도가 다른
    사진입니다. '물 확률'은 모델이 셀마다 믿는 값을 보여 줍니다. 주소창에 장면, 가구, 진입 지점, 보기가 저장되므로 링크를
    열면 같은 화면이 열립니다.`,
  // dynamic (app.js)
  'truck': '트럭', 'boat': '보트', 'on foot': '도보', 'no route': '경로 없음',
  'Truck can reach it': '트럭으로 도달 가능', 'Truck can reach it: confident': '트럭으로 도달 가능: 확실',
  'Boat needed': '보트 필요', 'No route: send an air team': '경로 없음: 항공 구조대 투입',
  'Probably reachable by truck: check first': '트럭으로 도달 가능성 있음: 먼저 확인',
  'truck, check first': '트럭, 먼저 확인', 'truck (confident)': '트럭 (확실)',
  'flooded building': '침수 건물', 'dry building': '미침수 건물', 'intact building': '온전한 건물', 'damaged building': '피해 건물',
  'Unreachable by road': '도로로 도달 불가', 'houses': '가구', 'by boat': '보트', 'no route ': '경로 없음',
  'more look truck-reachable: check first': '가구는 트럭 가능해 보임: 먼저 확인',
  'No houses in this photo': '이 사진에는 가구가 없습니다', 'Try the next photo: › (or the N key).': '다음 사진을 보세요: › (또는 N 키).',
  'No entry point in this photo: choose one': '이 사진에는 진입 지점이 없습니다: 직접 고르세요',
  'Tap a house to send a rescue team.': '가구를 눌러 구조대를 보내세요.',
  'Tap where the rescue team enters (road, yard or water).': '구조대가 진입할 곳을 누르세요 (도로, 마당 또는 물).',
  'That is a building: tap a road, a yard or water.': '건물입니다: 도로, 마당 또는 물을 누르세요.',
  'truck stops at': '트럭 정차 지점', 'point': '곳', 'points': '곳',
  'No truck, boat or walking route from the entry point on this map.': '이 지도에서는 진입 지점에서 트럭, 보트, 도보 경로가 없습니다.',
  'at the entry point': '진입 지점',
  'Link copied': '링크 복사됨', 'Copy the address bar': '주소창을 복사하세요',
  'House': '가구', 'Predicted': '예측', 'Label': '라벨',
  'risk-aware truck': '위험 회피 트럭', 'shortest truck': '최단 트럭',
  'road clear': '통행 가능 도로', 'road flooded': '침수 도로', 'building flooded': '침수 건물', 'building dry': '미침수 건물',
  'road blocked': '막힌 도로', 'building intact': '온전한 건물', 'building damaged': '피해 건물', 'water': '물',
  'Hurricane Harvey, Texas 2017': '허리케인 하비, 텍사스 2017', 'Hurricane Michael, Florida 2018': '허리케인 마이클, 플로리다 2018',
  'Route cost': '경로 비용', 'cells': '셀', 'no steps': '이동 없음',
  'Previous photo (P)': '이전 사진 (P)', 'Next photo (N)': '다음 사진 (N)',
  'Prediction and labelled map disagree': '예측과 라벨 지도가 다름',
  'Print briefing (map + house list)': '브리핑 인쇄 (지도 + 가구 목록)',
  'One page for a field team: the map as shown, with the route, and every house with its status, minutes and truck stop (prediction)':
    '현장 팀용 한 장: 지금 보이는 지도와 경로, 모든 가구의 상태, 소요 시간, 트럭 정차 지점 (예측)',
  'No houses in this photo.': '이 사진에는 가구가 없습니다.',
  'Truck stops': '트럭 정차', 'boat launch': '보트 출발', 'walk-on point': '도보 출발',
  // model switch
  'Model': '모델',
  'Chameleon as submitted: few-shot, plus 30 labelled photos for tuning': '제출한 Chameleon: 퓨샷, 조정용 라벨 사진 30장 추가',
  'DeepLabV3-ResNet50 fine-tuned on the same labelled photos (same-budget baseline)': '같은 라벨 사진으로 미세 조정한 DeepLabV3-ResNet50 (같은 예산 기준 모델)',
  'DeepLabV3 ensemble fine-tuned on the same labelled photos (thresholds tuned on the same 30 photos)':
    '같은 라벨 사진으로 미세 조정한 DeepLabV3 앙상블 (같은 조정용 사진 30장으로 임계값 조정)',
  'Not available for this photo': '이 사진에는 아직 없음',
  'models disagree most': '두 모델이 가장 다른 순',
  // printed dispatch plan
  'Dispatch plan': '출동 계획', 'Truck to the door': '트럭으로 집 앞까지', 'Truck stop': '트럭 정차',
  'No route from the entry point': '진입 지점에서 경로 없음',
};

const params = () => new URLSearchParams(location.hash.slice(1));

export const DEFAULT_LANG = (navigator.language || '').toLowerCase().startsWith('ko') ? 'ko' : 'en';
export let LANG = ['en', 'ko'].includes(params().get('lang')) ? params().get('lang') : DEFAULT_LANG;

export function t(s) {
  return LANG === 'ko' && KO[s] !== undefined ? KO[s] : s;
}

export function setLang(lang) {
  LANG = lang;
  document.documentElement.lang = lang;
  applyStatic();
}

/** translate the static page: data-i18n (text), data-i18n-html (HTML), data-i18n-attr="attr:key|attr:key" */
export function applyStatic() {
  for (const el of document.querySelectorAll('[data-i18n]')) {
    el.dataset.en ??= el.textContent;
    el.textContent = t(el.dataset.en);
  }
  for (const el of document.querySelectorAll('[data-i18n-html]')) {
    el.dataset.en ??= el.innerHTML;
    el.innerHTML = LANG === 'ko' && KO[el.dataset.i18nHtml] ? KO[el.dataset.i18nHtml] : el.dataset.en;
  }
  for (const el of document.querySelectorAll('[data-i18n-attr]')) {
    for (const pair of el.dataset.i18nAttr.split('|')) {
      const attr = pair.split(':')[0];
      const key = `en_${attr.replace(/-/g, '_')}`;
      el.dataset[key] ??= el.getAttribute(attr) || '';
      el.setAttribute(attr, t(el.dataset[key]));
    }
  }
}
