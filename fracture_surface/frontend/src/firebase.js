// fracture_surface/frontend/src/firebase.js

import { initializeApp } from "firebase/app";

import {
  getFirestore,
  collection,
  addDoc,
  getDocs,
  getDoc,
  setDoc,
  deleteDoc,
  doc,
  orderBy,
  query,
  limit,
  where,
  documentId,
} from "firebase/firestore";


// ==========================================
// 0. DB 저장 On/Off 스위치
// ==========================================
// true  → Firestore 실제로 읽기/쓰기 수행
// false → 네트워크 호출 없이 즉시 반환 (로컬 개발용, 빠름)
//
// 웹 화면 상단의 토글로 바꿀 수 있고, 선택한 값은 브라우저(localStorage)에 기억된다.
// DB_STORE_DEFAULT 는 처음 접속했을 때(저장된 값이 없을 때)의 기본값이다.

const DB_STORE_DEFAULT = true;
const DB_STORE_KEY = "fracture.dbStore";

function readStoredDbStore() {
  try {
    const saved = window.localStorage.getItem(DB_STORE_KEY);
    if (saved === null) return DB_STORE_DEFAULT;
    return saved === "true";
  } catch {
    return DB_STORE_DEFAULT; // 시크릿 모드 등 localStorage 를 못 쓰는 환경
  }
}

let DB_STORE = readStoredDbStore();

const dbStoreListeners = new Set();

/** 현재 DB 저장 On/Off 상태 */
export function isDbStoreEnabled() {
  return DB_STORE;
}

/** DB 저장 On/Off 변경 (웹 토글에서 호출) */
export function setDbStoreEnabled(enabled) {
  DB_STORE = Boolean(enabled);
  try {
    window.localStorage.setItem(DB_STORE_KEY, String(DB_STORE));
  } catch {
    // 저장 못 해도 이번 세션 동안은 적용된다
  }
  console.log(`[DB_STORE] ${DB_STORE ? "ON" : "OFF"}`);
  dbStoreListeners.forEach((listener) => listener(DB_STORE));
}

/** On/Off 변경 구독. 반환값을 호출하면 구독 해제 */
export function subscribeDbStore(listener) {
  dbStoreListeners.add(listener);
  return () => dbStoreListeners.delete(listener);
}


// ==========================================
// 1. Firebase 프로젝트 설정
// ==========================================

const firebaseConfig = {
  apiKey: "AIzaSyDUXgli766zJ_Iww8UKINX7tLbLkqFHiyE",
  authDomain: "grain-a489f.firebaseapp.com",
  projectId: "grain-a489f",
  storageBucket: "grain-a489f.firebasestorage.app",
  messagingSenderId: "604330631780",
  appId: "1:604330631780:web:fa3dc1fa808914e3a6b571",
  measurementId: "G-23LTZ2MHG6",
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);

const HISTORY_COLLECTION = "analysisHistory";

// 보고서는 분석 기록과 별도 컬렉션에 저장한다.
//  - 문서 ID = 분석 기록의 docId (1:1 연결, 조회가 단순)
//  - 기록 문서에는 base_image 등이 이미 들어 있어 1MB 한도에 가까울 수 있으므로 분리
const REPORT_COLLECTION = "analysisReports";


// ==========================================
// 2. Firestore 저장 데이터 변환 (기존과 동일)
// ==========================================

function sanitizeForFirestore(value) {
  if (value === undefined) return null;
  if (value === null) return null;

  if (Array.isArray(value)) {
    return value.map((item) => {
      if (Array.isArray(item)) {
        return {
          __nestedArray: true,
          values: sanitizeForFirestore(item),
        };
      }
      return sanitizeForFirestore(item);
    });
  }

  if (typeof value === "object") {
    const cleaned = {};
    Object.entries(value).forEach(([key, item]) => {
      cleaned[key] = sanitizeForFirestore(item);
    });
    return cleaned;
  }

  return value;
}


// ==========================================
// 3. Firestore 데이터 복원 (기존과 동일)
// ==========================================

function restoreFromFirestore(value) {
  if (Array.isArray(value)) {
    return value.map(restoreFromFirestore);
  }

  if (value && typeof value === "object") {
    if (value.__nestedArray === true) {
      return restoreFromFirestore(value.values);
    }

    const restored = {};
    Object.entries(value).forEach(([key, item]) => {
      restored[key] = restoreFromFirestore(item);
    });
    return restored;
  }

  return value;
}


// ==========================================
// 4. 분석 기록 저장
// ==========================================

export async function addHistoryItem(item) {

  if (!DB_STORE) {
    console.log("[DB_STORE=false] 저장 건너뜀");
    return null;
  }

  try {
    const cleanedItem = sanitizeForFirestore(item);

    const docRef = await addDoc(
      collection(db, HISTORY_COLLECTION),
      cleanedItem
    );

    console.log("Firestore 기록 저장 성공:", docRef.id);

    return docRef.id;

  } catch (error) {
    console.error("Firestore 기록 저장 실패:", error);
    throw error;
  }
}


// ==========================================
// 5. 최근 분석 기록 불러오기
// ==========================================

export async function fetchHistory() {

  if (!DB_STORE) {
    console.log("[DB_STORE=false] 불러오기 건너뜀");
    return [];
  }

  try {
    const q = query(
      collection(db, HISTORY_COLLECTION),
      orderBy("id", "desc"),
      limit(20)
    );

    const snapshot = await getDocs(q);

    const items = snapshot.docs.map((document) => {
      const data = restoreFromFirestore(document.data());
      return { ...data, docId: document.id };
    });

    console.log("Firestore 기록 불러오기 성공:", items.length);

    return items;

  } catch (error) {
    console.error("Firestore 기록 불러오기 실패:", error);
    throw error;
  }
}


// ==========================================
// 6. 분석 기록 전체 삭제
// ==========================================

export async function clearAllHistory() {

  if (!DB_STORE) {
    console.log("[DB_STORE=false] 삭제 건너뜀");
    return;
  }

  try {
    const snapshot = await getDocs(collection(db, HISTORY_COLLECTION));

    await Promise.all(
      snapshot.docs.map((document) =>
        deleteDoc(doc(db, HISTORY_COLLECTION, document.id))
      )
    );

    await clearAllReports();

    console.log("Firestore 전체 기록 삭제 완료");

  } catch (error) {
    console.error("Firestore 기록 삭제 실패:", error);
    throw error;
  }
}


// ==========================================
// 6-1. 선택한 분석 기록만 삭제
// ==========================================

/**
 * 선택한 분석 기록 삭제. 연결된 보고서(analysisReports/{docId})도 함께 지운다.
 * @param {string[]} docIds  analysisHistory 문서 ID 목록
 * @returns {number} 삭제한 기록 수 (DB_STORE=false 이면 0)
 */
export async function deleteHistoryItems(docIds) {

  if (!DB_STORE) {
    console.log("[DB_STORE=false] 선택 삭제 건너뜀");
    return 0;
  }

  const ids = (docIds || []).filter(Boolean);
  if (ids.length === 0) return 0;

  try {
    await Promise.all(
      ids.flatMap((id) => [
        deleteDoc(doc(db, HISTORY_COLLECTION, id)),
        deleteDoc(doc(db, REPORT_COLLECTION, id)), // 보고서가 없어도 오류 없음
      ])
    );

    console.log("Firestore 선택 기록 삭제 완료:", ids.length);
    return ids.length;

  } catch (error) {
    console.error("Firestore 선택 기록 삭제 실패:", error);
    throw error;
  }
}


// ==========================================
// 7. 보고서 저장 / 불러오기
// ==========================================

/**
 * 보고서 저장 (같은 기록에 다시 저장하면 덮어씀)
 * @param {string} historyDocId  analysisHistory 문서 ID
 * @param {object} report        /report 응답 (characteristics, basis, cause ...)
 * @param {object} extra         { gradcamSnapshot, prediction, confidence }
 */
export async function saveReport(historyDocId, report, extra = {}) {
  if (!DB_STORE) {
    console.log("[DB_STORE=false] 보고서 저장 건너뜀");
    return null;
  }
  if (!historyDocId) {
    throw new Error("연결할 분석 기록(docId)이 없습니다.");
  }

  const { status, ...reportBody } = report || {};

  const payload = sanitizeForFirestore({
    historyDocId,
    report: reportBody,
    gradcam_snapshot: extra.gradcamSnapshot || null,
    prediction: extra.prediction || null,
    confidence: extra.confidence || null,
    savedAt: Date.now(),
  });

  await setDoc(doc(db, REPORT_COLLECTION, historyDocId), payload);
  console.log("Firestore 보고서 저장 성공:", historyDocId);
  return payload.savedAt;
}

/** 기록 1건의 보고서 불러오기. 없으면 null */
export async function fetchReport(historyDocId) {
  if (!DB_STORE || !historyDocId) return null;

  const snap = await getDoc(doc(db, REPORT_COLLECTION, historyDocId));
  if (!snap.exists()) return null;

  return restoreFromFirestore(snap.data());
}

/** 목록에 '보고서' 배지를 붙이기 위해, 보고서가 있는 기록 docId 집합을 반환 */
export async function fetchReportIds(historyDocIds) {
  if (!DB_STORE) return new Set();

  const ids = (historyDocIds || []).filter(Boolean);
  const found = new Set();

  // Firestore 'in' 조건은 한 번에 최대 30개
  for (let i = 0; i < ids.length; i += 30) {
    const chunk = ids.slice(i, i + 30);
    const q = query(
      collection(db, REPORT_COLLECTION),
      where(documentId(), "in", chunk)
    );
    const snapshot = await getDocs(q);
    snapshot.docs.forEach((d) => found.add(d.id));
  }

  return found;
}

/** 보고서 전체 삭제 (기록 전체 삭제 시 함께 호출) */
export async function clearAllReports() {
  if (!DB_STORE) return;

  const snapshot = await getDocs(collection(db, REPORT_COLLECTION));
  await Promise.all(
    snapshot.docs.map((d) => deleteDoc(doc(db, REPORT_COLLECTION, d.id)))
  );
  console.log("Firestore 보고서 전체 삭제 완료");
}