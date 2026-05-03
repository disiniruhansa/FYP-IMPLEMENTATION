import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseFallbackConfig = {
  apiKey: "AIzaSyA3E2WEUOLgiHdytxZ8X7MlF9Lq7i3m1cM",
  authDomain: "empowerher-chatbot.firebaseapp.com",
  projectId: "empowerher-chatbot",
  storageBucket: "empowerher-chatbot.firebasestorage.app",
  messagingSenderId: "628213320602",
  appId: "1:628213320602:web:25e0eb336de4c5a2b58823",
};

const firebaseConfig = {
  apiKey:
    import.meta.env.VITE_FIREBASE_API_KEY || firebaseFallbackConfig.apiKey,
  authDomain:
    import.meta.env.VITE_FIREBASE_AUTH_DOMAIN ||
    firebaseFallbackConfig.authDomain,
  projectId:
    import.meta.env.VITE_FIREBASE_PROJECT_ID || firebaseFallbackConfig.projectId,
  storageBucket:
    import.meta.env.VITE_FIREBASE_STORAGE_BUCKET ||
    firebaseFallbackConfig.storageBucket,
  messagingSenderId:
    import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID ||
    firebaseFallbackConfig.messagingSenderId,
  appId: import.meta.env.VITE_FIREBASE_APP_ID || firebaseFallbackConfig.appId,
};

const requiredConfig = Object.values(firebaseConfig);

export const isFirebaseConfigured = requiredConfig.every(
  (value) => typeof value === "string" && value.trim().length > 0
);

const app = isFirebaseConfigured ? initializeApp(firebaseConfig) : null;

export const auth = app ? getAuth(app) : null;
export const db = app ? getFirestore(app) : null;
