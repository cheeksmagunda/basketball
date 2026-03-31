import styles from './Header.module.css';

export default function Header() {
  return (
    <header className={styles.header}>
      <img
        src="/oracle-ball.svg"
        alt="The Oracle"
        className={styles.logo}
        width={32}
        height={32}
      />
      <h1 className={styles.title}>THE ORACLE</h1>
    </header>
  );
}
