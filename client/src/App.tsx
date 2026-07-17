import { useState } from "react";
import { motion } from "framer-motion";
import { Header } from "./components/Header";
import { DigitSelector } from "./components/DigitSelector";
import { GeneratedImage } from "./components/GeneratedImage";
import { useGenerateDigit } from "./useGenerateDigit";

export default function App() {
    const [selectedDigit, setSelectedDigit] = useState<number | null>(null);
    const { data, isPending, isError, mutate } = useGenerateDigit();

    const handleSelect = (digit: number) => {
        setSelectedDigit(digit);
        mutate(digit);
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="min-h-screen flex flex-col items-center px-4 pt-12"
        >
            <Header />

            <motion.main
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="flex flex-col md:flex-row items-center gap-10"
            >
                <DigitSelector
                    selected={selectedDigit}
                    onSelect={handleSelect}
                />
                <GeneratedImage
                    image={data?.image ?? null}
                    isLoading={isPending}
                    isError={isError}
                    label={selectedDigit}
                />
            </motion.main>
        </motion.div>
    );
}
