import { motion } from "framer-motion";

const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const;

const buttonVariants = {
    idle: { scale: 1 },
    hover: { scale: 1.1 },
    tap: { scale: 0.95 },
};

interface DigitSelectorProps {
    selected: number | null;
    onSelect: (label: number) => void;
}

export function DigitSelector({ selected, onSelect }: DigitSelectorProps) {
    return (
        <div className="flex flex-col items-center gap-3">
            <h2 className="text-lg font-medium text-slate-300">
                Выберите цифру
            </h2>
            <div className="grid grid-cols-5 gap-2">
                {DIGITS.map((digit) => (
                    <motion.button
                        key={digit}
                        variants={buttonVariants}
                        initial="idle"
                        whileHover="hover"
                        whileTap="tap"
                        onClick={() => onSelect(digit)}
                        className={`
              w-12 h-12 rounded-lg text-xl font-bold transition-colors
              ${
                  selected === digit
                      ? "bg-primary text-white shadow-lg shadow-primary/30"
                      : "bg-surface-light text-slate-300 hover:bg-surface-lighter"
              }
            `}
                    >
                        {digit}
                    </motion.button>
                ))}
            </div>
        </div>
    );
}
