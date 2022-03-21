# myo_gan
- (주)인스페이스에서 인턴으로 활동하면서 김태영 기술이사님과 함께하는 프로젝트 입니다.
- 이 프로젝트의 궁극적 목표는 Myo센서만을 이용하여 손의 모양을(state 등)을 generate하는 것 입니다.
- Myo라는 armband의 데이터를 이용할 것입니다.
- DCGAN을 이용해 손 모양을 generate할 것 입니다.

## Plan
- 진행이 되는대로 계속해서 코드와 이미지, 결과 등을 올리겠습니다.
- Wiki에 개발일지를 정리해서 올리고 있습니다. 참고하셔도 될 것 같습니다.


## Contact me
- Name : Park sang-min
- email : jigeria@naver.com / jigeria114@gmail.com

### Memo
MYO 데이터를 수집하기 위해서 다음과 같은 작업이 필요합니다.
1. [이 링크](https://s3.amazonaws.com/thalmicdownloads/windows/SDK/myo-sdk-win-0.9.0.zip)를 눌러 MYO SDK를 다운받고 압축을 해제합니다.
2. source/emg_data_sample.cpp 파일을 SDK 폴더 하위의 samples/ 디렉토리에 덮어씌웁니다.
