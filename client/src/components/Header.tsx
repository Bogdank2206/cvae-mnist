import { motion } from 'framer-motion';

export function Header() {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 }}
      className="mb-10 text-center"
    >
      <h1 className="text-3xl font-bold text-white mb-2">
        CVAE Генератор
      </h1>
      <p className="text-slate-400">
        Условный вариационный автокодировщик для MNIST
      </p>
    </motion.header>
  );
}
