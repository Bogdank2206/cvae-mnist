import { motion, AnimatePresence } from "framer-motion";

interface ImageCardProps {
    image: string | null;
    isLoading: boolean;
    isError: boolean;
    label: number | null;
}

export function ImageCard({
    image,
    isLoading,
    isError,
    label,
}: ImageCardProps) {
    return (
        <div className="relative w-48 h-48 rounded-2xl bg-surface-light border border-slate-700 overflow-hidden flex items-center justify-center">
            <AnimatePresence mode="wait">
                {isLoading && (
                    <motion.div
                        key="loading"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="flex flex-col items-center gap-3"
                    >
                        <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                        <span className="text-sm text-slate-400">
                            Генерация...
                        </span>
                    </motion.div>
                )}

                {isError && (
                    <motion.p
                        key="error"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-sm text-red-400"
                    >
                        Ошибка генерации
                    </motion.p>
                )}

                {image && !isLoading && (
                    <motion.img
                        key={label}
                        src={image}
                        alt={`Сгенерированная цифра ${label}`}
                        className="w-28 h-28"
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.8 }}
                        transition={{
                            type: "spring",
                            stiffness: 200,
                            damping: 20,
                        }}
                    />
                )}
            </AnimatePresence>

            {label === null && !isLoading && (
                <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-sm text-slate-500"
                >
                    Выберите цифру
                </motion.p>
            )}
        </div>
    );
}
