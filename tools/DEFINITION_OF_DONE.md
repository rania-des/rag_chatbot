# Definition of Done (DoD) — Projet OMNIA

> Ce document définit les critères de « Done » (terminé) pour chaque User Story et fonctionnalité
> développée dans le cadre du projet OMNIA. Une fonctionnalité n'est considérée comme terminée
> que lorsque **tous** les critères ci-dessous sont satisfaits.

---

## 1. Code & Qualité

| Critère | Description |
|---------|-------------|
| ✅ Compilé sans erreurs | Le code TypeScript compile sans erreurs (`tsc --noEmit` passe) côté front et back |
| ✅ Linting passé | Aucune erreur ESLint bloquante dans les fichiers modifiés |
| ✅ Pas de `console.log` en production | Les logs de debug sont supprimés ou conditionnés par `NODE_ENV` |
| ✅ Code commenté | Les fonctions complexes et les modules sont documentés (JSDoc/TSDoc) |
| ✅ Pas de code mort | Les imports inutilisés et le code commenté sont supprimés |

---

## 2. Tests

| Critère | Description |
|---------|-------------|
| ✅ Tests unitaires | Les fonctions critiques (chiffrement, authentification, calculs de notes) ont des tests unitaires |
| ✅ Tests d'intégration | Les endpoints API principaux sont testés (routes CRUD, authentification) |
| ✅ Tests manuels UI | Chaque écran a été testé manuellement sur mobile (375px) et desktop (1280px) |
| ✅ Scénarios d'erreur | Les cas d'erreur (réseau, données vides, permissions) sont gérés et testés |

---

## 3. Sécurité

| Critère | Description |
|---------|-------------|
| ✅ Chiffrement AES-256-GCM | Les données sensibles (tokens, données personnelles en transit) sont chiffrées avec AES-256-GCM |
| ✅ Hachage des mots de passe | Les mots de passe sont hachés via bcrypt (géré par Supabase Auth) |
| ✅ Authentification JWT | Toutes les routes protégées vérifient le token JWT Bearer |
| ✅ Rate limiting | Les endpoints sensibles (login, AI, chat) ont un rate limiter actif |
| ✅ Validation des entrées | Les données utilisateur sont validées avec Zod côté backend |
| ✅ CORS configuré | Les origines autorisées sont explicitement listées dans la configuration |
| ✅ Variables d'environnement | Aucun secret n'est hardcodé — utilisation exclusive de `.env` |

---

## 4. Accessibilité (WCAG 2.1 AA)

| Critère | Description |
|---------|-------------|
| ✅ Navigation clavier | Tous les éléments interactifs sont accessibles via Tab / Enter / Escape |
| ✅ Focus visible | Un anneau de focus visible (`:focus-visible`) est affiché sur tous les éléments interactifs |
| ✅ Contrastes | Ratio de contraste minimum de 4.5:1 pour le texte normal, 3:1 pour le grand texte |
| ✅ ARIA landmarks | Les rôles ARIA (`role="main"`, `role="navigation"`, `role="banner"`) sont appliqués |
| ✅ Labels des formulaires | Tous les champs de formulaire ont un `<label>` associé ou un `aria-label` |
| ✅ Messages d'erreur | Les erreurs de formulaire sont annoncées via `role="alert"` et `aria-describedby` |
| ✅ Skip to content | Un lien « Aller au contenu principal » est disponible pour les utilisateurs clavier |
| ✅ Reduced motion | Les animations respectent `prefers-reduced-motion: reduce` |

---

## 5. Internationalisation (i18n)

| Critère | Description |
|---------|-------------|
| ✅ Support multilingue | L'interface supporte le Français (FR), l'Anglais (EN) et l'Arabe (AR) |
| ✅ RTL | Le mode Right-to-Left est correctement appliqué pour l'arabe (`dir="rtl"` sur `<html>`) |
| ✅ Clés de traduction | Aucune chaîne en dur dans les composants — toutes passent par `useTranslation()` ou `T[lang]` |
| ✅ Font arabe | La police Noto Sans Arabic est chargée pour le rendu correct de l'arabe |

---

## 6. Interface & UX

| Critère | Description |
|---------|-------------|
| ✅ Responsive | L'interface fonctionne sur mobile (375px+), tablette (768px+) et desktop (1280px+) |
| ✅ Dark mode | Le mode sombre est fonctionnel et cohérent sur tous les écrans |
| ✅ États de chargement | Un spinner ou skeleton est affiché pendant les chargements API |
| ✅ États vides | Un message clair est affiché quand il n'y a pas de données |
| ✅ Gestion d'erreur UI | Les erreurs réseau/API affichent un message utilisateur avec option de réessai |

---

## 7. Déploiement

| Critère | Description |
|---------|-------------|
| ✅ Build production | `npm run build` passe sans erreurs côté front et back |
| ✅ Variables d'environnement | Toutes les variables `.env` sont documentées dans `.env.example` |
| ✅ Migration BDD | Les migrations Supabase sont documentées et réversibles |
| ✅ Health check | L'endpoint `/health` retourne un statut OK |

---

## 8. Documentation

| Critère | Description |
|---------|-------------|
| ✅ README à jour | Le README reflète l'état actuel du projet (installation, lancement, architecture) |
| ✅ API documentée | Les endpoints API sont documentés (méthode, route, body, response) |
| ✅ Changelog | Les changements significatifs sont documentés dans le commit message |

---

## Processus de validation

```
1. Développeur → Auto-revue selon la DoD ci-dessus
2. Développeur → Push + Merge Request
3. Product Owner → Validation fonctionnelle
4. Déploiement → Vérification en staging
5. ✅ User Story marquée comme « Done »
```

---

*Dernière mise à jour : Avril 2026 — Projet OMNIA v1.0.0*
