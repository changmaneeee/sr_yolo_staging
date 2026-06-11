# 07-05. Arch5b 실험

## 개요

Arch5b의 학습/평가는 **4090 서버에서 진행**. 본 staging repo는 결과만 정리.

## RFDN Arch5b (SOTA)

- Phase 2 (warmup) + Phase 3 (joint) 학습 완료
- subset6418 mAP@50 = **0.9102**
- TRT FP16 latency: 12ms (4060)
- Small vessel recall: +84% (Phase 2 → Phase 3)
- Best learnable α 값: (서버 결과 참조)

## DRCT Arch5b

- 4090 서버에서 Phase 2/3 학습 (현재 진행/완료)
- subset6418 mAP@50: (서버에서 가져와야 함)
- Phase 3 checkpoint 위치: (서버 경로)

## HAT / MAN Arch5b

- 시간 여유 시 진행 예정
- 현재는 RFDN, DRCT만 main result

## 서버에서 가져와야 할 자료

자세한 명령은 [docs/PROMPTS/server_arch5_request.md](../PROMPTS/server_arch5_request.md) 참조.

- Phase 2 checkpoint
- Phase 3 checkpoint
- 학습 로그 (loss curve)
- 평가 결과 JSON
- 학습 config

## 공정성 보장

- 평가는 본 staging repo에서 같은 방식으로 (eval_arch.py + subset6418)
- 학습은 서버지만, train set만 사용 (val 누설 없음)
- Phase 3 checkpoint를 받아서 본 repo에서 re-evaluate 가능

## 검증 포인트

- [ ] Phase 2/Phase 3 학습이 train set만 사용
- [ ] 평가는 subset6418에서 동일
- [ ] 결과가 메인 표에 반영
- [ ] 서버 코드와 본 repo 코드가 일치
