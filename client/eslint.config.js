import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactPlugin from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";

export default tseslint.config(
    // Глобальные игноры
    {
        ignores: ["dist"],
    },
    // Базовые JS-правила
    js.configs.recommended,
    // React
    {
        files: ["**/*.{ts,tsx}"],
        plugins: {
            react: reactPlugin,
        },
        settings: {
            react: {
                version: "detect",
            },
        },
        rules: {
            "react/react-in-jsx-scope": "off",
            "react/prop-types": "off",
        },
    },
    // TypeScript
    ...tseslint.configs.recommended,
    {
        files: ["**/*.{ts,tsx}"],
        rules: {
            "@typescript-eslint/no-unused-vars": [
                "warn",
                { argsIgnorePattern: "^_" },
            ],
            "@typescript-eslint/consistent-type-imports": [
                "error",
                { prefer: "type-imports" },
            ],
        },
    },
    // React Hooks
    {
        files: ["**/*.{ts,tsx}"],
        plugins: {
            "react-hooks": reactHooks,
        },
        rules: reactHooks.configs.recommended.rules,
    },
    // React Refresh (Vite HMR)
    {
        files: ["**/*.{ts,tsx}"],
        plugins: {
            "react-refresh": reactRefresh,
        },
        rules: {
            "react-refresh/only-export-components": [
                "warn",
                { allowConstantExport: true },
            ],
        },
    },
    // Глобальные переменные (браузер)
    {
        languageOptions: {
            globals: globals.browser,
        },
    },
);
