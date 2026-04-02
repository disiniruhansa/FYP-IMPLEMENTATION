import { useState } from "react";
import type { User } from "firebase/auth";

type AuthMode = "signin" | "signup";

type AuthPanelProps = {
  authBusy: boolean;
  authEnabled: boolean;
  authReady: boolean;
  historyReady: boolean;
  user: User | null;
  onEmailAuth: (
    mode: AuthMode,
    email: string,
    password: string
  ) => Promise<void>;
  onGoogleAuth: () => Promise<void>;
  onSignOut: () => Promise<void>;
};

export function AuthPanel({
  authBusy,
  authEnabled,
  authReady,
  historyReady,
  user,
  onEmailAuth,
  onGoogleAuth,
  onSignOut,
}: AuthPanelProps) {
  const [mode, setMode] = useState<AuthMode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  const submitDisabled =
    authBusy || !email.trim() || password.trim().length < 6 || !authReady;

  if (!authEnabled) {
    return (
      <div className="card authCard">
        <h2>Save Chat History</h2>
        <p className="muted">
          Firebase is not configured in the frontend yet. Add the Vite Firebase
          variables to enable sign in and saved conversations.
        </p>
      </div>
    );
  }

  if (user) {
    return (
      <div className="card authCard">
        <h2>Account</h2>
        <p className="authUserLabel">{user.email ?? "Signed in with Google"}</p>
        <p className="muted">
          This account can reopen the saved conversation on the next visit.
        </p>
        <div className="authActions">
          <button
            className="primaryBtn wide"
            onClick={() => void onSignOut()}
            disabled={authBusy}
          >
            Sign Out
          </button>
          <div className="authHint">
            {historyReady
              ? "History sync is active."
              : "Loading saved conversation..."}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card authCard">
      <div className="authHeader">
        <h2>Save Chat History</h2>
        <div className="authToggle">
          <button
            className={mode === "signin" ? "authTab active" : "authTab"}
            onClick={() => setMode("signin")}
            type="button"
          >
            Sign In
          </button>
          <button
            className={mode === "signup" ? "authTab active" : "authTab"}
            onClick={() => setMode("signup")}
            type="button"
          >
            Sign Up
          </button>
        </div>
      </div>

      <p className="muted">
        Guest mode still works. Sign in only if you want the chatbot to remember
        your conversation.
      </p>

      <label className="authField">
        <span>Email</span>
        <input
          type="email"
          value={email}
          onChange={(event) => setEmail(event.target.value)}
          placeholder="name@example.com"
          autoComplete="email"
        />
      </label>

      <label className="authField">
        <span>Password</span>
        <input
          type="password"
          value={password}
          onChange={(event) => setPassword(event.target.value)}
          placeholder="At least 6 characters"
          autoComplete={mode === "signup" ? "new-password" : "current-password"}
        />
      </label>

      <div className="authActions">
        <button
          className="primaryBtn wide"
          onClick={() => void onEmailAuth(mode, email, password)}
          disabled={submitDisabled}
          type="button"
        >
          {authBusy
            ? "Please wait..."
            : mode === "signup"
              ? "Create Account"
              : "Sign In"}
        </button>
        <button
          className="ghost wide"
          onClick={() => void onGoogleAuth()}
          disabled={authBusy || !authReady}
          type="button"
        >
          Continue with Google
        </button>
      </div>
    </div>
  );
}
