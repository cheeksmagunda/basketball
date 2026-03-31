import styles from './Header.module.css';

export default function Header() {
  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>🏀</div>
        <div>
          <h1 className={styles.title}>THE ORACLE</h1>
          <span className={styles.sub}>Sees what others miss</span>
        </div>
      </div>
    </header>
  );
}
