import type { User } from "firebase/auth";
import {
  GoogleAuthProvider,
  createUserWithEmailAndPassword,
  onAuthStateChanged,
  signInWithEmailAndPassword,
  signInWithPopup,
  signOut,
} from "firebase/auth";
import { auth } from "../lib/firebase";

function requireAuth() {
  if (!auth) {
    throw new Error("Firebase Auth is not configured.");
  }

  return auth;
}

export function subscribeToAuthChanges(callback: (user: User | null) => void) {
  return onAuthStateChanged(requireAuth(), callback);
}

export async function signUpWithEmail(email: string, password: string) {
  await createUserWithEmailAndPassword(requireAuth(), email, password);
}

export async function signInWithEmail(email: string, password: string) {
  await signInWithEmailAndPassword(requireAuth(), email, password);
}

export async function signInWithGoogle() {
  const provider = new GoogleAuthProvider();
  await signInWithPopup(requireAuth(), provider);
}

export async function signOutUser() {
  await signOut(requireAuth());
}
