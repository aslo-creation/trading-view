# Mettre le terminal en ligne (gratuit) — Streamlit Community Cloud

## Pourquoi pas Vercel ?
Vercel exécute des fonctions éphémères (Next.js, API serverless). Streamlit
est un serveur Python permanent avec WebSocket : incompatible. Si un jour le
frontend est migré en Next.js, Vercel hébergera ce frontend — mais le backend
Python devra toujours vivre ailleurs (Render, Fly.io…).

## Étapes (≈ 20 minutes)

### 1. Mettre le code sur GitHub (dépôt PRIVÉ)
1. Créez un compte sur github.com, puis installez **GitHub Desktop**
   (desktop.github.com) — pas besoin de ligne de commande.
2. Ouvrez le fichier `.gitignore` du projet et **supprimez la ligne**
   `config/auth_config.yaml` (elle ne contient que des hash bcrypt ; c'est
   acceptable dans un dépôt privé, jamais dans un dépôt public).
   Le fichier `.env` reste ignoré : il ne part JAMAIS sur GitHub.
3. GitHub Desktop → *Add local repository* → choisissez le dossier
   `quant-terminal` → *Publish repository* → **cochez "Keep this code
   private"** → Publish.

### 2. Déployer sur Streamlit Community Cloud
1. Allez sur **share.streamlit.io**, connectez-vous avec votre compte GitHub.
2. *New app* → choisissez votre dépôt `quant-terminal`, branche `main`,
   fichier principal `app.py` → *Deploy*.

### 3. Configurer les secrets (remplace votre .env local)
Dans l'app déployée : menu **⋮ → Settings → Secrets**, collez :

    SESSION_SIGNING_KEY = "votre_clé_de_64_caractères_hex"
    ANTHROPIC_API_KEY = "sk-ant-..."        # optionnel (Comité IA)
    FRED_API_KEY = "..."                    # optionnel (macro réelle)

Sauvegardez : l'app redémarre toute seule. Le code lit ces secrets
automatiquement (pont st.secrets → variables d'environnement dans app.py).

### 4. Vérifications une fois en ligne
- L'URL est publique : c'est VOTRE mur d'authentification qui protège tout.
  Choisissez un mot de passe long ; 5 échecs = verrouillage 5 minutes.
- Onglet 🔐 Admin : confirmez que les clés s'affichent MASQUÉES.
- Si des symboles passent en "DÉMO" en ligne alors qu'ils étaient LIVE en
  local : Yahoo limite parfois les adresses IP partagées du cloud. Remède :
  bouton Rafraîchir un peu plus tard, ou intervalle d'auto-refresh à 5-10 min.

## Limites du gratuit (à connaître)
- L'app s'endort après ~15 min sans visite ; le premier visiteur suivant
  attend ~30-60 s qu'elle se réveille. Vos réglages en mémoire (watchlist
  ajoutée, dernier débrief) sont remis à zéro à chaque réveil.
- Ressources modestes (1 Go RAM) : largement suffisant pour cet usage.

## Alternatives équivalentes
- **Hugging Face Spaces** (gratuit, Streamlit natif, secrets = variables
  d'environnement — zéro changement de code).
- **Render.com** (gratuit avec mise en veille ; commande de démarrage :
  `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`).
