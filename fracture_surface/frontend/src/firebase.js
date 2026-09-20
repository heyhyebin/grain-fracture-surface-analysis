// fracture_surface/frontend/src/firebase.js
import { initializeApp } from "firebase/app";
import {
  getFirestore,
  collection,
  addDoc,
  getDocs,
  deleteDoc,
  doc,
  orderBy,
  query,
  limit,
} from "firebase/firestore";

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyDUXgli766zJ_Iww8UKINX7tLbLkqFHiyE",
  authDomain: "grain-a489f.firebaseapp.com",
  projectId: "grain-a489f",
  storageBucket: "grain-a489f.firebasestorage.app",
  messagingSenderId: "604330631780",
  appId: "1:604330631780:web:fa3dc1fa808914e3a6b571",
  measurementId: "G-23LTZ2MHG6"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);

const HISTORY_COLLECTION = "analysisHistory";

// 기록 저장
export async function addHistoryItem(item) {
  await addDoc(collection(db, HISTORY_COLLECTION), item);
}

// 최근 기록 불러오기 (최대 20개)
export async function fetchHistory() {
  const q = query(
    collection(db, HISTORY_COLLECTION),
    orderBy("id", "desc"),
    limit(20)
  );
  const snapshot = await getDocs(q);
  return snapshot.docs.map((d) => ({ docId: d.id, ...d.data() }));
}

// 기록 전체 삭제
export async function clearAllHistory() {
  const snapshot = await getDocs(collection(db, HISTORY_COLLECTION));
  await Promise.all(
    snapshot.docs.map((d) => deleteDoc(doc(db, HISTORY_COLLECTION, d.docId || d.id)))
  );
}